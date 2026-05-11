from typing_extensions import TypedDict, Any
import pandas as pd
from destination_filter.preferences import filter_destinasi_bds_style_whom_async
from destination_filter.planner import pemilihan_titik_wisata
from routing.routing_core import routing_destinasi
from destination_filter.get_nearby_food_hotel import get_top_food_hotel
from utils import generate_metadata

async def generate_itinerary(data_input, data_rating_modified, jumlah_hari, whom, styles, budget, daerah_tujuan_destinasi):
    # Ubah data dari supabase (list of dict) ke dataframe
    data_input = pd.DataFrame(data_input)
    data_rating_modified = pd.DataFrame(data_rating_modified)

    data_destinasi_rating_modified = data_rating_modified.loc[(data_input["category"] != "Hotels & Accomodations") &
                    (data_input["category"] != "Food & Drink") &
                    (data_input["poi"] != "event")]

    data_destinasi = data_input.loc[(data_input["category"] != "Hotels & Accomodations") &
                    (data_input["category"] != "Food & Drink") &
                    (data_input["poi"] != "event")]

    data_hotel_restoran = data_input.loc[((data_input["category"] == "Hotels & Accomodations") |
                          (data_input["category"] == "Food & Drink")) &
                          (data_input["poi"] != "event")]

    # Get preferences
    time_limit, data_destinasi = await filter_destinasi_bds_style_whom_async(data_destinasi, jumlah_hari, whom, styles)
    data_destinasi_rating_modified = data_destinasi_rating_modified.loc[data_destinasi.index]

    # Search destination
    result, total_biaya = await pemilihan_titik_wisata(data_destinasi_rating_modified, data_destinasi, jumlah_hari, budget, time_limit, False)

    # Destination routing
    titik_hasil_TSP_dibagi_per_hari, total_dist = await routing_destinasi(data_destinasi, result, jumlah_hari)

    # Find nearest accommodation
    restoran, hotel = await get_top_food_hotel(data_hotel_restoran, data_destinasi, titik_hasil_TSP_dibagi_per_hari, jumlah_hari, 5)


    titik_hasil_TSP_dibagi_per_hari = generate_metadata(titik_hasil_TSP_dibagi_per_hari, data_destinasi)
    restoran = generate_metadata(restoran, data_hotel_restoran)
    hotel = generate_metadata(hotel, data_hotel_restoran)

    def hitung_rata_rata(
            input_data: dict[int, dict[str, dict[str, Any]]]
    ) -> int:
        total = count = 0
        for day in input_data.values():
            for place in day.values():
                total += place["price_mean"]
                count += 1

        input_rata_rata = int(total / count) if count else 0
        return input_rata_rata

    harga_rerata_restoran = hitung_rata_rata(restoran)
    harga_rerata_hotel = hitung_rata_rata(hotel)

    return titik_hasil_TSP_dibagi_per_hari, restoran, hotel, total_biaya, harga_rerata_restoran, harga_rerata_hotel