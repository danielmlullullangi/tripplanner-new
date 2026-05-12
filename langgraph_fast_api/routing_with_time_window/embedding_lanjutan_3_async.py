"""
embedding_lanjutan_3_async.py — VRPTW Edition
===============================================
Perubahan dari versi TSP:

1. TripPlannerState — tambah dua field baru:
   - arrival_times_per_hari : Dict[int, List[int]]
     Waktu tiba di setiap destinasi per hari, dalam menit sejak tengah malam.
     Diisi oleh select_destination_food_hotel dari return generate_itinerary.

   - waktu_mulai_perjalanan : int
     Opsional — jam mulai perjalanan dalam menit. Saat ini tidak diekspos
     ke user (hardcoded di preferences.py), tapi bisa dikembangkan
     sebagai input parameter.

2. select_destination_food_hotel:
   Sebelum : P, food, hotel, total_destination_cost, food_mean, hotel_mean = result
   Sesudah : P, arrival_times, food, hotel, total_destination_cost, food_mean, hotel_mean = result
   → unpack nilai arrival_times dari return generate_itinerary yang baru
   → simpan ke state['arrival_times_per_hari']

3. parafrase_output:
   Tambahkan field 'jam_tiba' ke setiap item destinasi dalam output akhir.
   Format: "HH:MM" string agar mudah ditampilkan di frontend.
   Helper function menit_ke_jam_str diimpor dari tsp_solver.py.
"""

from typing import Dict, Any, Optional, List, Literal
from typing_extensions import TypedDict


# =============================================================================
# STATE DEFINITION — tambah dua field baru
# =============================================================================

class TripPlannerState(TypedDict):
    """
    Mencakup semua informasi yang diperlukan sepanjang pipeline.

    Field baru untuk VRPTW:
    - arrival_times_per_hari : waktu tiba (menit) per destinasi per hari
    - waktu_mulai_perjalanan : jam mulai wisata (menit, default 480 = 08:00)
    """
    destination_name: str
    num_days: int
    user_prompt: str
    price_max: int
    travel_whom: str
    destination_type: List[str]

    data: Optional[List[Dict[str, Any]]]
    data_enhanced: Optional[List[Dict[str, Any]]]
    user_prompt_embed: List[float]
    guardrail_input: bool

    rekomendasi_destinasi: Dict[int, List[str]]
    rekomendasi_makanan: Dict[int, Dict[str, Dict[str, Any]]]
    rekomendasi_hotel: Dict[int, Dict[str, Dict[str, Any]]]

    # BARU: waktu tiba di setiap destinasi, dalam menit sejak tengah malam
    # Contoh: {1: [480, 600, 720], 2: [480, 570, 660]}
    arrival_times_per_hari: Dict[int, List[int]]

    estimasi_biaya: int
    rerata_harga_makanan: int
    rerata_harga_hotel: int
    rekomendasi_destinasi_paraph: List[Dict[int, List[List]]]


# =============================================================================
# MODEL EMBEDDING (tidak berubah)
# =============================================================================

from openai import OpenAI

openai_api_key_embed = "tripplanner_embed"
openai_api_base_embed = "http://llm_llama_cpp:6000/v1"
client_embed = OpenAI(
    api_key=openai_api_key_embed,
    base_url=openai_api_base_embed,
)

def model():
    models_embed = client_embed.models.list()
    model_embed = models_embed.data[0].id
    return model_embed


# =============================================================================
# ROUTER NODES (tidak berubah)
# =============================================================================

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


# =============================================================================
# EMBED PROMPT (tidak berubah)
# =============================================================================

import asyncio

async def embed_prompt(state: TripPlannerState) -> TripPlannerState:
    try:
        print(">>> EMBEDDING INPUT")
        loop = asyncio.get_running_loop()
        responses_embed = await loop.run_in_executor(
            None,
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


# =============================================================================
# DATABASE (tidak berubah)
# =============================================================================

import asyncpg
import json
import pandas as pd
from decimal import Decimal
from datetime import datetime, date
import copy

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

async def fetch_filter_dan_sort_data(state: TripPlannerState) -> TripPlannerState:
    if not state.get('user_prompt_embed'):
        print("ERROR: user_prompt_embed kosong.")
        return state

    pool = await get_db_pool()
    sql = """
    SELECT title, body, type, url_image, brief_description, price_min, 
    price_max, type_poi, howtoget_there, tips, phone, address, website_sosmed, latitude, longitude,
    provinces, destinations, must_see, counter, poi, category, duration, rating, rating_count,
    link_drive, rating_total, price_mean
    FROM public.trip_planner 
    WHERE destinations = $1 AND poi != 'event'
    ORDER BY vektor <=> $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, state['destination_name'], str(state['user_prompt_embed']))

    state['data'] = list(map(convert_row_fast, rows))
    return state


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def ubah_rating_data(state: TripPlannerState) -> TripPlannerState:
    try:
        print(">>> UBAH RATING")
        data = state.get('data')
        if not data:
            return state

        enhanced_data = copy.deepcopy(data)
        for i in range(min(20, len(enhanced_data))):
            rating_val = enhanced_data[i].get('rating_total')
            if rating_val is not None and isinstance(rating_val, (int, float, Decimal)):
                if isinstance(rating_val, Decimal):
                    rating_val = float(rating_val)
                enhanced_data[i]['rating_total'] = rating_val + 3
            else:
                enhanced_data[i]['rating_total'] = 3

        state['data_enhanced'] = enhanced_data
        return state
    except Exception as e:
        print(f"Error: {e}")
        return state


# =============================================================================
# SELECT DESTINATION — PERUBAHAN: unpack arrival_times dari return baru
# =============================================================================

from main import generate_itinerary

async def select_destination_food_hotel(state: TripPlannerState) -> TripPlannerState:
    """
    Perubahan dari versi TSP:
    Return generate_itinerary sekarang 7 nilai (sebelumnya 6).
    Nilai baru: arrival_times_per_hari di posisi ke-2 (index 1).
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
            state['destination_name'],
        )

        # PERUBAHAN: unpack 7 nilai (sebelumnya 6)
        # Urutan: destinasi, arrival_times, food, hotel, biaya, food_mean, hotel_mean
        (
            P,
            arrival_times,        # BARU
            food,
            hotel,
            total_destination_cost,
            food_mean,
            hotel_mean,
        ) = result

        state['rekomendasi_destinasi'] = P
        state['arrival_times_per_hari'] = arrival_times   # BARU
        state['rekomendasi_makanan'] = food
        state['rekomendasi_hotel'] = hotel
        state['estimasi_biaya'] = total_destination_cost
        state['rerata_harga_makanan'] = food_mean
        state['rerata_harga_hotel'] = hotel_mean

        return state

    except Exception as e:
        print(e)
        return state


# =============================================================================
# PARAFRASE OUTPUT — PERUBAHAN: tambah jam_tiba ke setiap item destinasi
# =============================================================================

# Import helper konversi menit → string jam dari tsp_solver
from routing.tsp_solver import menit_ke_jam_str

import time

def parafrase_output(state: TripPlannerState) -> TripPlannerState:
    """
    Perubahan dari versi TSP:
    Setiap item destinasi sekarang memiliki field 'jam_tiba' tambahan.
    Field ini berisi string waktu tiba ("HH:MM") yang diambil dari
    arrival_times_per_hari yang baru tersedia di state.

    Jika arrival_times tidak tersedia (fallback), jam_tiba diisi string kosong
    agar kompatibilitas ke belakang tetap terjaga.

    Contoh output per destinasi:
    {
        "nama": "Kawah Putih",
        "jam_tiba": "09:10",          ← BARU
        "body": "...",
        "tips": "...",
        "latitude": -7.16,
        "longitude": 107.39,
        "howtoget_there": "..."
    }
    """
    try:
        paraphrased_destinations = {}
        for day, destinations in state["rekomendasi_destinasi"].items():
            paraphrased_destinations[day] = {}
            for nama, detail in destinations.items():
                paraphrased_destinations[day][nama] = detail

        paraphrased_foods = {}
        for day, foods in state["rekomendasi_makanan"].items():
            paraphrased_foods[day] = {}
            for nama, detail in foods.items():
                paraphrased_foods[day][nama] = detail

        paraphrased_hotels = {}
        for day, hotels in state["rekomendasi_hotel"].items():
            paraphrased_hotels[day] = {}
            for nama, detail in hotels.items():
                paraphrased_hotels[day][nama] = detail

        # Ambil arrival_times dari state (bisa kosong jika fallback)
        arrival_times_per_hari: Dict[int, List[int]] = state.get(
            "arrival_times_per_hari", {}
        )

        result_list = []
        num_days = state.get("num_days", 0)

        for day in range(1, num_days + 1):
            makanan_list = []
            for nama, detail in state["rekomendasi_makanan"].get(day, {}).items():
                makanan_list.append({
                    nama: {
                        "latitude": detail.get("latitude"),
                        "longitude": detail.get("longitude"),
                    }
                })

            hotel_list = []
            for nama, detail in state["rekomendasi_hotel"].get(day, {}).items():
                hotel_list.append({
                    nama: {
                        "latitude": detail.get("latitude"),
                        "longitude": detail.get("longitude"),
                    }
                })

            day_data = {
                "hari": day,
                "makanan_terdekat": makanan_list,
                "hotel_terdekat": hotel_list,
                "destinasi": [],
            }

            dest_hari = paraphrased_destinations.get(day, {})

            # Ambil arrival_times untuk hari ini
            arrival_hari: List[int] = arrival_times_per_hari.get(day, [])

            for idx, (nama, detail) in enumerate(dest_hari.items()):
                # Ambil jam tiba untuk destinasi ke-idx hari ini
                # Jika arrival_times tidak tersedia, jam_tiba dikosongkan
                if idx < len(arrival_hari):
                    jam_tiba_str = menit_ke_jam_str(arrival_hari[idx])
                else:
                    jam_tiba_str = ""

                dest_item = {
                    "nama": nama,
                    "jam_tiba": jam_tiba_str,       # BARU
                    "body": detail.get("body", ""),
                    "tips": detail.get("tips", ""),
                    "latitude": detail.get("latitude", ""),
                    "longitude": detail.get("longitude", ""),
                    "howtoget_there": detail.get("howtoget_there", ""),
                }
                day_data["destinasi"].append(dest_item)

            result_list.append(day_data)

        state['rekomendasi_destinasi_paraph'] = result_list
        return state

    except Exception as e:
        print(f"Error in parafrase_output: {e}")
        return state


# =============================================================================
# LANGGRAPH GRAPH (tidak berubah — node dan edge sama persis)
# =============================================================================

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
graph.add_edge("select_destination_food_hotel", "parafrase_output")
graph.add_edge("parafrase_output", END)

agent = graph.compile()