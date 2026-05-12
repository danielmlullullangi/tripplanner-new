"""
preferences.py — VRPTW Edition
================================
Perubahan dari versi TSP:

1. Return value bertambah satu:
   Sebelum : (time_limit, data_terfilter)
   Sesudah : (time_limit_menit, waktu_mulai_per_hari, data_terfilter)

2. time_limit sekarang dalam MENIT (bukan jam):
   Sebelum : time_limit = [8, 7, 9]  → jam
   Sesudah : time_limit = [480, 420, 540]  → menit
   Alasan  : solver VRPTW bekerja dalam domain menit. Konversi
             dilakukan di sini agar main.py dan planner.py tidak
             perlu melakukan konversi masing-masing.

3. Tambah waktu_mulai_per_hari:
   List waktu mulai perjalanan per hari dalam menit (e.g. [480, 480, 480]).
   Diteruskan ke routing_core.routing_destinasi sebagai parameter baru.
   Secara konsep: jam berapa wisatawan mulai dari hotel setiap harinya.
   Untuk saat ini diasumsikan sama setiap hari (08:00), tapi bisa
   dikustomisasi per hari di masa depan.

Catatan kompatibilitas dengan planner.py:
   planner.py menggunakan time_limit dalam konteks:
       sum(time_limit) * 100   ← sebagai batas total waktu dalam CP-SAT
   Sebelumnya sum([8,7,9]) = 24 (jam), lalu × 100 = 2400.
   Sekarang sum([480,420,540]) = 1440 (menit), lalu × 100 = 144000.
   Karena planner.py meng-scale semua nilai ke rentang 0-100 dengan
   fungsi scaler(), perubahan satuan ini aman — relasi proporsional
   antar nilai tetap terjaga. Tidak perlu mengubah planner.py.
"""

import pandas as pd
import numpy as np
import asyncio


# Waktu mulai perjalanan default dalam menit sejak tengah malam.
# 480 = jam 08:00 pagi.
WAKTU_MULAI_DEFAULT: int = 8 * 60  # 480 menit


def filter_destinasi_bds_style_whom(
        data_terfilter: pd.DataFrame,
        jumlah_hari: int,
        whom: str = "default",
        styles: str | list[str] = ["default"],
) -> tuple[list[int], list[int], pd.DataFrame]:
    """
    Memfilter data destinasi berdasarkan gaya perjalanan dan tipe wisatawan,
    serta menentukan batas waktu harian dan jam mulai perjalanan.

    Parameters
    ----------
    data_terfilter : pd.DataFrame
        Data destinasi yang akan difilter.
    jumlah_hari : int
        Jumlah hari perjalanan.
    whom : str
        Tipe wisatawan: "solo", "family", "couple", "friends", "elderly", default.
    styles : str | list[str]
        Gaya wisata: "nature", "cultural", "classic", "cityscape", "historical", "default".

    Returns
    -------
    time_limit_menit : list[int]
        Durasi wisata per hari dalam MENIT (bukan jam seperti versi lama).
        Digunakan oleh planner.py sebagai constraint total durasi di CP-SAT,
        dan oleh routing_core.py sebagai parameter kapasitas waktu solver.

    waktu_mulai_per_hari : list[int]
        Jam mulai perjalanan per hari dalam menit sejak tengah malam.
        Untuk saat ini seragam (08:00 = 480 menit) untuk semua hari.
        Diteruskan ke routing_core.routing_destinasi.

    data_terfilter : pd.DataFrame
        DataFrame setelah difilter berdasarkan styles.
    """

    # ------------------------------------------------------------------
    # BAGIAN 1 — Tentukan durasi wisata per hari berdasarkan whom
    # ------------------------------------------------------------------
    # Perubahan: nilai dihasilkan langsung dalam menit (× 60 dari versi lama).
    # random.randint dipertahankan untuk variasi alami antar request.

    if whom == "solo":
        # Solo traveler: energik, bisa wisata 8-9 jam
        time_limit_menit = [
            np.random.randint(low=8 * 60, high=9 * 60)
            for _ in range(jumlah_hari)
        ]

    elif whom == "family":
        # Family: perlu istirahat anak, 6-8 jam
        time_limit_menit = [
            np.random.randint(low=6 * 60, high=8 * 60)
            for _ in range(jumlah_hari)
        ]

    elif whom == "couple":
        # Couple: santai tapi bisa lama, 9-11 jam
        time_limit_menit = [
            np.random.randint(low=9 * 60, high=11 * 60)
            for _ in range(jumlah_hari)
        ]

    elif whom == "friends":
        # Friends: dinamis, 7-10 jam
        time_limit_menit = [
            np.random.randint(low=7 * 60, high=10 * 60)
            for _ in range(jumlah_hari)
        ]

    elif whom == "elderly":
        # Elderly: perlu lebih banyak istirahat, 5-7 jam
        time_limit_menit = [
            np.random.randint(low=5 * 60, high=7 * 60)
            for _ in range(jumlah_hari)
        ]

    else:
        # Default: fleksibel, 7-11 jam
        time_limit_menit = [
            np.random.randint(low=7 * 60, high=11 * 60)
            for _ in range(jumlah_hari)
        ]

    # ------------------------------------------------------------------
    # BAGIAN 2 — Tentukan waktu mulai perjalanan per hari (BARU)
    # ------------------------------------------------------------------
    # Saat ini semua hari mulai jam 08:00.
    # Di masa depan bisa berbeda per hari atau per whom
    # (misalnya elderly mulai lebih siang: 09:00).
    waktu_mulai_per_hari: list[int] = [
        WAKTU_MULAI_DEFAULT for _ in range(jumlah_hari)
    ]

    # ------------------------------------------------------------------
    # BAGIAN 3 — Filter berdasarkan travel style (TIDAK BERUBAH)
    # ------------------------------------------------------------------
    def choose_style(
            df: pd.DataFrame,
            styles: str | list[str] = ["default"]
    ) -> pd.DataFrame:
        if styles is None or "default" in styles:
            return df

        if isinstance(styles, str):
            styles = [styles]

        style_map = {
            "cultural": [
                "Classes & Workshops",
                "Concerts & Shows",
                "Sights & Landmarks",
            ],
            "classic": [
                "Museums",
                "Sights & Landmarks",
                "Classes & Workshops",
                "Concerts & Shows",
            ],
            "nature": [
                "Nature & Parks",
                "Zoos & Aquarium",
                "Outdoor Activities",
                "Boat Tours & Water Sports",
            ],
            "cityscape": [
                "Sights & Landmarks",
                "Shopping",
                "Nightlife",
            ],
            "historical": [
                "Museums",
                "Sights & Landmarks",
            ],
        }

        style_dipilih: set[str] = set()
        for style in styles:
            style_dipilih.update(style_map.get(style, []))

        return df[df["category"].isin(style_dipilih)]

    data_terfilter = choose_style(data_terfilter, styles)

    return time_limit_menit, waktu_mulai_per_hari, data_terfilter


async def filter_destinasi_bds_style_whom_async(
        data_terfilter: pd.DataFrame,
        jumlah_hari: int,
        whom: str = "default",
        styles: str | list[str] = ["default"],
) -> tuple[list[int], list[int], pd.DataFrame]:
    """
    Async wrapper — tidak ada perubahan logika,
    hanya meneruskan return baru (3 nilai) dari versi sync.
    """
    return await asyncio.to_thread(
        filter_destinasi_bds_style_whom,
        data_terfilter,
        jumlah_hari,
        whom,
        styles,
    )
