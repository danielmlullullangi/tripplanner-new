from typing import Dict, Any, Optional, List, Literal
from typing_extensions import TypedDict

class TripPlannerState(TypedDict):
    """
    Mencakup semua informasi yang diperlukan sepanjang pipeline
    """
    destination_name: str                                       #Daerah destinasi (kota)
    num_days: int                                               #Budget
    user_prompt: str                                            #Kustomisasi prompt dari user
    price_max: int
    travel_whom: str                                            #Tipe perjalanan (solo, family, couple, friends, elderly)
    destination_type: List[str]                                 #Tipe destinasi (Nature & Parks, Museums, Sights & Landmarks, 
                                                                #Classes & Workshops, Shopping, Zoos & Aquariums, Water & Amusement Parks,
                                                                #Spas & Wellness, Boat Tours & Water Sports, Fun & Games, Shopping,
                                                                #Outdoor Activities, dan Nightlife)                   
                                                                #Di sini, tidak ada pilihan Food & Drink dan Hotels & Accomodations, 
                                                                #jadi otomatis direkomendasikan. Pilihan Concerts & Shows tidak dimasukkan.
    data: Optional[List[Dict[str, Any]]]                        #Data yang diambil dari database
    data_enhanced: Optional[List[Dict[str, Any]]]               #Data yang telah disesuaikan nilai ratingnya
    user_prompt_embed: List[float]                              #Hasil embedding user prompt
    guardrail_input: bool                                       #Hasil input guardrail
    rekomendasi_destinasi: Dict[int, List[str]]                 #Rencana perjalanan per hari yang telah diurutkan perjalanannya
    rekomendasi_makanan: Dict[int, Dict[str, Dict[str, Any]]]   #Rekomendasi tempat makan per hari beserta dengan metadata
    rekomendasi_hotel: Dict[int, Dict[str, Dict[str, Any]]]     #Rekomendasi hotel per hari beserta dengan metadata
    estimasi_biaya: int                                         #Estimasi biaya yang dihasilkan
    rerata_harga_makanan: int                                   #Harga rata-rata makanan hasil rekomendasi
    rerata_harga_hotel: int                                     #Harga rata-rata hotel hasil rekomendasi
    rekomendasi_destinasi_paraph: List[Dict[int, List[List]]]   #Rencana perjalanan per hari beserta deskripsi dan tips yang telah di
                                                                #parafrase oleh AI

"""
Model untuk Embedding
"""
from openai import OpenAI

openai_api_key_embed = "tripplanner_embed"
# Kalau llama cpp di WSL dan langgraph di windows: pakai IP WSL 172.30.252.174
# Kalau keduanya di WSL atau Windows: pakai localhost
# Kalau langgraph di WSL dan llama cpp di windows: 10.251.106.105 (IP internet, ipconfig)
openai_api_base_embed = "http://llm_llama_cpp:6000/v1" # IP Windows
client_embed = OpenAI(
    api_key=openai_api_key_embed,
    base_url=openai_api_base_embed,
)

def model():
    models_embed = client_embed.models.list()
    model_embed = models_embed.data[0].id
    return model_embed

def input_router(state: TripPlannerState) -> Literal["guardrail_input", "fetch_filter_dan_sort_data"]:
    try:
        print(">>> INPUT ROUTER")
        if state.get("user_prompt", "").strip():
            return "guardrail_input"
        else:
            return "fetch_filter_dan_sort_data"
    except Exception as e:
        print(e)
        return state


def guardrail_input(state: TripPlannerState) -> TripPlannerState:
    try:
        print(">>> GUARDRAIL INPUT")
        state["guardrail_input"] = True
        return state
    except Exception as e:
        return state

def router_after_guardrail(state: TripPlannerState) -> Literal["embed_prompt", "__end__"]:
    try:
        print(">>> ROUTER AFTER GUARDRAIL")
        if state.get("guardrail_input"):
            return "embed_prompt"
        else:
            return "__end__"
    except Exception as e:
        print(e)
        return state

async def embed_prompt(state: TripPlannerState) -> TripPlannerState:
    """
    Melakukan embedding terhadap input prompt dari user
    """
    try:
        print(">>> EMBEDDING INPUT")
        # Jalankan fungsi sinkron di thread pool agar tidak memblokir event loop
        loop = asyncio.get_running_loop()
        responses_embed = await loop.run_in_executor(
            None,  # default executor
            lambda: client_embed.embeddings.create(
                input=[state['user_prompt']],
                model=model(),
            )
        )

        state['user_prompt_embed'] = responses_embed.data[0].embedding

        return state
    except Exception as e:
        print(e)
        return state



import asyncpg, asyncio
import json
import pandas as pd
from decimal import Decimal

_db_pool = None
_db_pool_lock = asyncio.Lock()
async def init_db_pool():
    global _db_pool
    async with _db_pool_lock:
        if _db_pool is not None:
            return
        _db_pool = await asyncpg.create_pool(
            host="46.250.232.129",
            port=5436,
            database="postgres",
            user="postgres",
            password="QIC4YTWePq6zoDOp8xXjPjR99T7ctUluVvA0mG93KnSytYWE9lO6mduTs43JYE4H",
            min_size=2,
            max_size=5,
            ssl=False,
            command_timeout=60,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
        )

async def get_db_pool():
    if _db_pool is None:
        await init_db_pool()
    return _db_pool

async def close_db_pool():
    global _db_pool
    if _db_pool:
        await _db_pool.close()

_NUMERIC_TYPES = (Decimal,)
def convert_row_fast(row):
    return {
        k: float(v) if isinstance(v, _NUMERIC_TYPES) else v
        for k, v in row.items()
    }

#Fetching yang sudah disesuaikan dengan Guvicorn
async def fetch_filter_dan_sort_data(state: TripPlannerState) -> TripPlannerState:
    """Mengambil data dari database menggunakan connection pool."""
    if not state.get('user_prompt_embed'):
        print("ERROR: user_prompt_embed kosong, tidak bisa melakukan query.")
        return state

    pool = await get_db_pool()  # dapatkan pool (dibuat jika belum ada)

    sql = """
    SELECT title, body, type, url_image, brief_description, price_min, 
    price_max, type_poi, howtoget_there, tips, phone, address, website_sosmed, latitude, longitude,
    provinces, destinations, must_see, counter, poi, category, duration, rating, rating_count, link_drive, rating_total, price_mean
    FROM public.trip_planner 
    WHERE destinations = $1 AND poi != 'event'
    ORDER BY vektor <=> $2
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, state['destination_name'], str(state['user_prompt_embed']))
        
    state['data'] = list(map(convert_row_fast, rows)) #data_list
    return state

import json
from decimal import Decimal
from datetime import datetime, date
import copy

#Custom encoder untuk menangani Decimal dan datetime
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)                               # Mengubah Decimal ke float
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()                          # Mengubah datetime/date ke string ISO
        return super().default(obj)


def ubah_rating_data(state: TripPlannerState) -> TripPlannerState:
    try:
        print(">>> UBAH RATING")
        data = state.get('data')
        if not data:
            return state

        # Buat salinan dalam untuk data_enhanced
        enhanced_data = copy.deepcopy(data)

        # Naikkan rating 20 pertama di enhanced_data
        for i in range(min(20, len(enhanced_data))):
            rating_val = enhanced_data[i].get('rating_total')
            if rating_val is not None and isinstance(rating_val, (int, float, Decimal)):
                if isinstance(rating_val, Decimal):
                    rating_val = float(rating_val)
                enhanced_data[i]['rating_total'] = rating_val + 3
            else:
                enhanced_data[i]['rating_total'] = 3 

        # Simpan ke state
        state['data_enhanced'] = enhanced_data   # data dengan rating dimodifikasi

        return state
    except Exception as e:
        print(f"Error: {e}")
        return state

from routing_planner_main import generate_itinerary
import asyncio

async def select_destination_food_hotel(state: TripPlannerState) -> TripPlannerState:
    """
    Melakukan pencarian destinasi, pencarian tempat makan dan akomodasi.
    Melakukan optimisasi rute juga menghasilkan metadata yang dibutuhkan.
    """
    try:
        print(">>> PILIH DESTINASI, OPTIMISASI RUTE, dan PENCARIAN HOTEL DAN MAKAN")

        result = await generate_itinerary(
            state['data'],
            state['data_enhanced'],
            state['num_days'],
            state['travel_whom'],
            state['destination_type'],
            state['price_max'],
            state['destination_name']
        )

        P, food, hotel, total_destination_cost, food_mean, hotel_mean = result

        print("tes2")
        state['rekomendasi_destinasi'] = P
        state['rekomendasi_makanan'] = food
        state['rekomendasi_hotel'] = hotel
        state['estimasi_biaya'] = total_destination_cost
        state['rerata_harga_makanan'] = food_mean
        state['rerata_harga_hotel'] = hotel_mean

        return state

    except Exception as e:
        print(e)
        return state

def parafrase_output(state: TripPlannerState) -> TripPlannerState:
    try:
        # print("H1")
        # 1. Format destinasi
        paraphrased_destinations = {}
        for day, destinations in state["rekomendasi_destinasi"].items():
            # print(day, destinations)
            paraphrased_destinations[day] = {}
            for nama, detail in destinations.items():
                paraphrased_destinations[day][nama] = detail
        # print("H2")
        # 2. Parafrase makanan (hasil disimpan untuk keperluan lain jika diperlukan)
        paraphrased_foods = {}
        for day, foods in state["rekomendasi_makanan"].items():
            paraphrased_foods[day] = {}
            for nama, detail in foods.items():
                paraphrased_foods[day][nama] = detail
        # print("H3")
        # 3. Parafrase hotel
        paraphrased_hotels = {}
        for day, hotels in state["rekomendasi_hotel"].items():
            paraphrased_hotels[day] = {}
            for nama, detail in hotels.items():
                paraphrased_hotels[day][nama] = detail

        # print("H4")
        # 4. Bangun struktur output per hari
        result_list = []
        num_days = state.get("num_days", 0)
        for day in range(1, num_days + 1):
            # Ambil data makanan per hari dengan latitude & longitude
            makanan_list = []
            for nama, detail in state["rekomendasi_makanan"].get(day, {}).items():
                makanan_list.append({
                    nama: {
                        "latitude": detail.get("latitude"),
                        "longitude": detail.get("longitude")
                    }
                })

            # (Opsional) Jika hotel juga ingin format serupa
            hotel_list = []
            for nama, detail in state["rekomendasi_hotel"].get(day, {}).items():
                hotel_list.append({
                    nama: {
                        "latitude": detail.get("latitude"),
                        "longitude": detail.get("longitude")
                    }
                })

            day_data = {
                "hari": day,
                "makanan_terdekat": makanan_list,
                "hotel_terdekat": hotel_list,   # jika diinginkan
                "destinasi": []
            }

            # Ambil destinasi dari hasil parafrase untuk hari ini
            dest_hari = paraphrased_destinations.get(day, {})
            for nama, detail in dest_hari.items():
                dest_item = {
                    "nama": nama,
                    "body": detail.get("body", ""),
                    "tips": detail.get("tips", ""),
                    "latitude": detail.get("latitude", ""),
                    "longitude": detail.get("longitude", ""),
                    "howtoget_there": detail.get("howtoget_there", "")
                }
                day_data["destinasi"].append(dest_item)
            result_list.append(day_data)

        print("H5")
        state['rekomendasi_destinasi_paraph'] = result_list

        return state

    except Exception as e:
        print(f"Error in parafrase_output: {e}")
        return state





from langgraph.graph import StateGraph, START, END

graph = StateGraph(TripPlannerState)

graph.add_node("guardrail_input", guardrail_input)
graph.add_node("embed_prompt", embed_prompt)
graph.add_node("fetch_filter_dan_sort_data", fetch_filter_dan_sort_data)
graph.add_node("ubah_rating_data", ubah_rating_data)
graph.add_node("select_destination_food_hotel", select_destination_food_hotel)
graph.add_node("parafrase_output", parafrase_output)

graph.add_conditional_edges(START, input_router)
graph.add_conditional_edges("guardrail_input", router_after_guardrail)
graph.add_edge("embed_prompt", "fetch_filter_dan_sort_data")
graph.add_edge("fetch_filter_dan_sort_data", "ubah_rating_data")
graph.add_edge("ubah_rating_data", "select_destination_food_hotel")
# graph.add_edge("select_destination_food_hotel", END)
graph.add_edge("select_destination_food_hotel", "parafrase_output")
graph.add_edge("parafrase_output", END)

agent = graph.compile()