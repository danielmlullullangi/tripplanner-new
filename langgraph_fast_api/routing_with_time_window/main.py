"""
main.py — VRPTW Edition
=========================
Perubahan dari versi TSP:

1. get_preferences:
   Sebelum : time_limit, data_destinasi = ...
   Sesudah : time_limit, waktu_mulai_per_hari, data_destinasi = ...
   → unpack nilai baru waktu_mulai_per_hari dari preferences.py

2. routing_destinasi:
   Sebelum : titik_hasil_TSP_dibagi_per_hari, total_dist = await routing_destinasi(...)
   Sesudah : destinasi_per_hari, arrival_times_per_hari, total_waktu = await routing_destinasi(
                 ..., waktu_mulai_menit=waktu_mulai_per_hari[0]
             )
   → teruskan waktu_mulai ke routing agar time window mulai dari jam yang benar
   → unpack 3 nilai return (sebelumnya 2)
   → arrival_times_per_hari adalah dict baru yang menyimpan jam tiba

3. generate_metadata:
   Dipanggil untuk destinasi_per_hari (nama baru dari titik_hasil_TSP_dibagi_per_hari).
   Nama variabel diubah agar jelas tidak ada kebingungan dengan versi TSP.

4. Return value generate_itinerary bertambah satu:
   Sebelum : (titik_hasil_TSP, restoran, hotel, total_biaya, food_mean, hotel_mean)
   Sesudah : (destinasi_per_hari, arrival_times_per_hari, restoran, hotel,
              total_biaya, food_mean, hotel_mean)
   → arrival_times_per_hari diteruskan ke embedding_lanjutan_3_async.py
     untuk disimpan di state dan ditampilkan ke user sebagai jadwal jam tiba.

Catatan:
   get_preferences sebelumnya dipanggil tanpa definisi fungsi di main.py —
   ini sepertinya seharusnya memanggil filter_destinasi_bds_style_whom_async.
   Import telah diperbaiki dan fungsi dipanggil secara eksplisit.
"""

from typing import Any
import pandas as pd

from destination_filter.preferences import filter_destinasi_bds_style_whom_async
from destination_filter.planner import trip_planner_selection
from routing.routing_core import routing_destinasi
from destination_filter.get_nearby_food_hotel import get_top_food_hotel
from utils import generate_metadata


async def generate_itinerary(
        data_input,
        data_rating_modified,
        jumlah_hari: int,
        whom: str,
        styles: list[str],
        budget: int,
        daerah_tujuan_destinasi: str,
) -> tuple[
    dict[int, dict],          # destinasi_per_hari (dengan metadata)
    dict[int, list[int]],     # arrival_times_per_hari — BARU
    dict,                     # restoran (dengan metadata)
    dict,                     # hotel (dengan metadata)
    int,                      # total_biaya
    int,                      # harga_rerata_restoran
    int,                      # harga_rerata_hotel
]:
    """
    Orkestrasi seluruh pipeline itinerari dari data mentah hingga output final.

    Perubahan utama dari versi TSP:
    - Unpack 3 nilai dari preferences (tambah waktu_mulai_per_hari)
    - Teruskan waktu_mulai ke routing_destinasi
    - Unpack 3 nilai dari routing_destinasi (tambah arrival_times_per_hari)
    - Return arrival_times_per_hari ke caller

    Parameters
    ----------
    data_input : list[dict]
        Data mentah dari database (semua kategori: destinasi, hotel, restoran).
    data_rating_modified : list[dict]
        Data dengan rating yang telah di-boost 20 teratas (semantic relevance).
    jumlah_hari : int
        Jumlah hari perjalanan.
    whom : str
        Tipe wisatawan (solo/family/couple/friends/elderly).
    styles : list[str]
        Gaya wisata (nature/cultural/classic/cityscape/historical/default).
    budget : int
        Anggaran maksimum dalam Rupiah.
    daerah_tujuan_destinasi : str
        Nama kota tujuan (untuk logging/debugging).

    Returns
    -------
    Lihat docstring fungsi di atas untuk detail setiap return value.
    """

    # ------------------------------------------------------------------
    # Langkah 1 — Konversi ke DataFrame dan pisahkan berdasarkan kategori
    # ------------------------------------------------------------------
    data_input = pd.DataFrame(data_input)
    data_rating_modified = pd.DataFrame(data_rating_modified)

    mask_bukan_destinasi = (
        (data_input["category"] == "Hotels & Accomodations") |
        (data_input["category"] == "Food & Drink") |
        (data_input["poi"] == "event")
    )

    data_destinasi_rating_modified = data_rating_modified.loc[~mask_bukan_destinasi]
    data_destinasi = data_input.loc[~mask_bukan_destinasi]
    data_hotel_restoran = data_input.loc[
        ((data_input["category"] == "Hotels & Accomodations") |
         (data_input["category"] == "Food & Drink")) &
        (data_input["poi"] != "event")
    ]

    # ------------------------------------------------------------------
    # Langkah 2 — Get preferences
    # Perubahan: unpack 3 nilai (tambah waktu_mulai_per_hari)
    # ------------------------------------------------------------------
    time_limit_menit, waktu_mulai_per_hari, data_destinasi = (
        await filter_destinasi_bds_style_whom_async(
            data_destinasi, jumlah_hari, whom, styles
        )
    )
    # Sinkronkan data rating modified dengan index data_destinasi yang sudah difilter
    data_destinasi_rating_modified = data_destinasi_rating_modified.loc[data_destinasi.index]

    # ------------------------------------------------------------------
    # Langkah 3 — Seleksi destinasi optimal (CP-SAT solver — tidak berubah)
    # ------------------------------------------------------------------
    result, total_biaya = await trip_planner_selection(
        data_destinasi_rating_modified,
        data_destinasi,
        jumlah_hari,
        budget,
        time_limit_menit,   # sekarang dalam menit, tapi planner.py aman (pakai scaler)
        False,
    )

    # ------------------------------------------------------------------
    # Langkah 4 — Routing VRPTW
    # Perubahan:
    #   - Tambah parameter waktu_mulai_menit
    #   - Unpack 3 return nilai (sebelumnya 2)
    #   - Nama variabel lebih eksplisit
    # ------------------------------------------------------------------
    # Gunakan waktu mulai hari pertama sebagai referensi.
    # Untuk multi-hari, setiap hari bisa punya waktu mulai sendiri —
    # tapi routing_destinasi saat ini memproses semua destinasi sekaligus
    # lalu membaginya. Waktu mulai hari pertama dipakai sebagai anchor.
    waktu_mulai_hari_ini = waktu_mulai_per_hari[0] if waktu_mulai_per_hari else 8 * 60

    destinasi_per_hari, arrival_times_per_hari, total_waktu_tempuh = (
        await routing_destinasi(
            data_destinasi,
            result,
            jumlah_hari,
            waktu_mulai_menit=waktu_mulai_hari_ini,   # parameter BARU
        )
    )

    # ------------------------------------------------------------------
    # Langkah 5 — Cari restoran dan hotel terdekat (tidak berubah)
    # ------------------------------------------------------------------
    restoran, hotel = await get_top_food_hotel(
        data_hotel_restoran,
        data_destinasi,
        destinasi_per_hari,
        jumlah_hari,
        5,
    )

    # ------------------------------------------------------------------
    # Langkah 6 — Enrichment metadata (tidak berubah)
    # ------------------------------------------------------------------
    destinasi_per_hari = generate_metadata(destinasi_per_hari, data_destinasi)
    restoran = generate_metadata(restoran, data_hotel_restoran)
    hotel = generate_metadata(hotel, data_hotel_restoran)

    # ------------------------------------------------------------------
    # Langkah 7 — Hitung harga rata-rata (tidak berubah)
    # ------------------------------------------------------------------
    def hitung_rata_rata(
            input_data: dict[int, dict[str, dict[str, Any]]]
    ) -> int:
        total = count = 0
        for day in input_data.values():
            for place in day.values():
                total += place.get("price_mean", 0)
                count += 1
        return int(total / count) if count else 0

    harga_rerata_restoran = hitung_rata_rata(restoran)
    harga_rerata_hotel = hitung_rata_rata(hotel)

    # ------------------------------------------------------------------
    # Return — arrival_times_per_hari ditambahkan sebagai nilai ke-2
    # ------------------------------------------------------------------
    return (
        destinasi_per_hari,
        arrival_times_per_hari,   # BARU
        restoran,
        hotel,
        total_biaya,
        harga_rerata_restoran,
        harga_rerata_hotel,
    )