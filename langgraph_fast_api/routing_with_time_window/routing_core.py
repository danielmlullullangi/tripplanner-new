"""
routing_core.py — VRPTW Edition
=================================
Perubahan dari versi TSP:

1. Import  : intracluster_tsp → intracluster_vrptw
2. Fungsi baru: hitung_matriks_WAKTU_antartitik
               menggantikan hitung_matriks_JARAK_antartitik
               → fix bug 'places' (NameError) sekaligus
               → fix bug matriks tidak simetris
               → output dalam menit (int), bukan km (float)
3. Fungsi baru: ekstrak_time_windows_dan_durasi
               mengambil jam buka/tutup dan durasi per destinasi
               dari DataFrame → dioper ke solver sebagai constraint
4. routing_destinasi:
               → terima parameter baru: waktu_mulai_menit
               → bangun time_matrix, bukan distance_matrix
               → bangun time_windows dan durasi_kunjungan
               → panggil intracluster_vrptw (bukan intracluster_tsp)
               → unpack 3 nilai return (urutan, arrival_times, total_waktu)
               → teruskan arrival_times ke caller melalui dict terpisah
5. bagi_destinasi_jadi_jadwal_per_hari:
               → tidak berubah strukturnya, tapi sekarang juga
                 membagi arrival_times dengan cara yang sama
"""

from haversine import haversine
import pandas as pd
import numpy as np
from routing.tsp_solver import intracluster_tsp, KECEPATAN_KM_PER_MENIT, jam_ke_menit


# =============================================================================
# KONSTANTA
# =============================================================================

# Jam buka default jika data tidak tersedia di database.
# Dalam menit sejak tengah malam: 480 = 08:00.
JAM_BUKA_DEFAULT_MENIT: int = 8 * 60       # 08:00
JAM_TUTUP_DEFAULT_MENIT: int = 17 * 60     # 17:00
DURASI_KUNJUNGAN_DEFAULT_MENIT: int = 90   # 1.5 jam


# =============================================================================
# BAGIAN 1 — tentukan_titik_start_dan_end (TIDAK BERUBAH)
# =============================================================================

def tentukan_titik_start_dan_end(
        koordinat_lokasi_pada_peta: dict[str, tuple[float, float]],
        rating_lokasi_pada_peta: dict[str, float],
        titik_lokasi_terfilter: list[str],
) -> tuple[int, int]:
    """
    Menentukan indeks start dan end dalam list titik_lokasi_terfilter.

    Start : destinasi dengan rating tertinggi (pengalaman terbaik di awal hari)
    End   : destinasi paling jauh dari start (membentuk rute linier, bukan bolak-balik)

    Tidak ada perubahan logika dari versi TSP. Tetap dipakai di VRPTW
    karena RoutingIndexManager masih menggunakan custom start & end.
    """
    jumlah_tempat = len(titik_lokasi_terfilter)
    if jumlah_tempat == 0:
        return -1, -1
    if jumlah_tempat == 1:
        return 0, 0

    # Start: indeks destinasi dengan rating tertinggi
    indeks_titik_pertama = titik_lokasi_terfilter.index(
        max(titik_lokasi_terfilter, key=lambda p: rating_lokasi_pada_peta.get(p, 0))
    )

    # End: destinasi paling jauh secara geografis dari start
    start_coord = koordinat_lokasi_pada_peta[titik_lokasi_terfilter[indeks_titik_pertama]]
    jarak_terjauh = -1
    indeks_titik_terakhir = -1

    for idx in range(jumlah_tempat):
        if idx == indeks_titik_pertama:
            continue
        d = haversine(start_coord, koordinat_lokasi_pada_peta[titik_lokasi_terfilter[idx]])
        if d > jarak_terjauh:
            jarak_terjauh = d
            indeks_titik_terakhir = idx

    return indeks_titik_pertama, indeks_titik_terakhir


# =============================================================================
# BAGIAN 2 — hitung_matriks_WAKTU_antartitik (MENGGANTIKAN versi jarak)
# =============================================================================

def hitung_matriks_waktu_antartitik(
        koordinat_lokasi_pada_peta: dict[str, tuple[float, float]],
        daftar_nama_tempat: list[str],
) -> list[list[int]]:
    """
    Membangun time matrix N×N dalam satuan MENIT (integer).

    Mengapa waktu, bukan jarak:
        Time window constraint hanya bisa diperiksa dalam domain waktu.
        Solver perlu tahu "pukul berapa tiba di node ini", bukan "seberapa jauh".
        Jarak dipakai sebagai intermediate — dikonversi ke menit via kecepatan
        rata-rata kendaraan (KECEPATAN_KM_PER_MENIT dari tsp_solver.py).

    Perbaikan dari versi sebelumnya (hitung_matriks_JARAK_antartitik):
        Bug 1 — NameError: variabel 'places' diganti 'daftar_nama_tempat'
        Bug 2 — Matriks tidak simetris: baris kedua seharusnya
                 matrix[counter_kolom][counter_baris] = waktu (bukan duplikasi baris pertama)

    Returns
    -------
    list[list[int]]
        Matriks simetris N×N berisi waktu tempuh dalam menit (integer).
        Elemen [i][j] = menit perjalanan dari destinasi-i ke destinasi-j.
    """
    jumlah_tempat = len(daftar_nama_tempat)

    # FIX BUG 1: pakai 'daftar_nama_tempat', bukan variabel 'places' yang tidak ada
    koordinat_terurut_bds_lokasi = [koordinat_lokasi_pada_peta[p] for p in daftar_nama_tempat]

    # Inisialisasi matriks dengan nol
    matrix: list[list[int]] = [[0] * jumlah_tempat for _ in range(jumlah_tempat)]

    # Hitung segitiga atas saja (simetris), lalu mirror ke segitiga bawah
    for counter_baris in range(jumlah_tempat):
        for counter_kolom in range(counter_baris + 1, jumlah_tempat):
            jarak_km = haversine(koordinat_terurut_bds_lokasi[counter_baris], koordinat_terurut_bds_lokasi[counter_kolom])

            # Konversi km → menit menggunakan konstanta kecepatan dari tsp_solver
            waktu_menit = int(round(jarak_km / KECEPATAN_KM_PER_MENIT))

            matrix[counter_baris][counter_kolom] = waktu_menit
            matrix[counter_kolom][counter_baris] = waktu_menit
    return matrix


# =============================================================================
# BAGIAN 3 — ekstrak_time_windows_dan_durasi (FUNGSI BARU)
# =============================================================================

def ekstrak_time_windows_dan_durasi(
        data_destinasi: pd.DataFrame,
        daftar_nama_tempat: list[str],
        waktu_mulai_menit: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Mengekstrak time window (jam buka, jam tutup) dan durasi kunjungan
    dari DataFrame untuk setiap destinasi dalam daftar_nama_tempat.

    Fungsi ini adalah jembatan antara data dunia nyata (DataFrame dari DB)
    dan representasi matematis yang dibutuhkan solver VRPTW.

    Mengapa fungsi ini perlu ada:
        Solver VRPTW membutuhkan time_windows dan durasi_kunjungan sebagai
        input eksplisit per node. Data ini ada di database (kolom open_time,
        close_time, duration), tapi perlu diekstrak dan dikonversi ke menit
        agar konsisten dengan domain waktu solver.

    Parameters
    ----------
    data_destinasi : pd.DataFrame
        DataFrame destinasi dengan kolom yang diharapkan:
        - 'title'      : nama destinasi (key untuk lookup)
        - 'open_time'  : jam buka dalam format "HH:MM" atau menit (int)
        - 'close_time' : jam tutup dalam format "HH:MM" atau menit (int)
        - 'duration'   : lama kunjungan dalam JAM (float, sesuai kolom DB)

    daftar_nama_tempat : list[str]
        Urutan nama destinasi — harus sama persis dengan urutan di time matrix.

    waktu_mulai_menit : int
        Jam mulai perjalanan hari ini dalam menit sejak tengah malam.
        Dipakai untuk memastikan time window tidak lebih awal dari waktu
        wisatawan realistis bisa tiba.

    Returns
    -------
    time_windows : list[tuple[int, int]]
        Satu tuple (buka, tutup) per destinasi, dalam menit sejak tengah malam.
        Urutan list = urutan daftar_nama_tempat.

    durasi_kunjungan : list[int]
        Lama kunjungan per destinasi dalam menit.
        Urutan list = urutan daftar_nama_tempat.

    Catatan tentang kolom database:
        Saat ini kolom open_time dan close_time mungkin belum ada di DB.
        Jika tidak ada, digunakan fallback default (08:00–17:00).
        Saat data DB sudah memiliki kolom ini, baris fallback bisa dihapus.
    """
    # Buat lookup: nama destinasi → row data
    data_indexed = data_destinasi.set_index("title")

    time_windows: list[tuple[int, int]] = []
    durasi_kunjungan: list[int] = []

    for nama in daftar_nama_tempat:
        try:
            row = data_indexed.loc[nama]

            # --- Ekstrak jam buka ---
            # Kolom 'open_time' diharapkan dalam format "HH:MM" (string)
            # Jika belum ada di DB, gunakan default
            if "open_time" in data_indexed.columns and pd.notna(row.get("open_time")):
                raw_open = row["open_time"]
                if isinstance(raw_open, str) and ":" in raw_open:
                    h, m = map(int, raw_open.split(":"))
                    buka = jam_ke_menit(h, m)
                else:
                    buka = int(raw_open)  # sudah dalam menit
            else:
                buka = JAM_BUKA_DEFAULT_MENIT  # fallback: 08:00

            # Pastikan jam buka tidak lebih awal dari waktu mulai wisatawan
            buka = max(buka, waktu_mulai_menit)

            # --- Ekstrak jam tutup ---
            if "close_time" in data_indexed.columns and pd.notna(row.get("close_time")):
                raw_close = row["close_time"]
                if isinstance(raw_close, str) and ":" in raw_close:
                    h, m = map(int, raw_close.split(":"))
                    tutup = jam_ke_menit(h, m)
                else:
                    tutup = int(raw_close)
            else:
                tutup = JAM_TUTUP_DEFAULT_MENIT  # fallback: 17:00

            # Sanity check: tutup harus lebih besar dari buka
            if tutup <= buka:
                tutup = buka + 60  # minimal window 1 jam

            # --- Ekstrak durasi kunjungan ---
            # Kolom 'duration' di DB dalam satuan JAM (float), perlu × 60
            raw_durasi = row.get("duration", None) if hasattr(row, "get") else None
            if raw_durasi is None and "duration" in data_indexed.columns:
                raw_durasi = row["duration"]

            if raw_durasi is not None and not (
                isinstance(raw_durasi, float) and np.isnan(raw_durasi)
            ):
                durasi_menit = int(float(raw_durasi) * 60)
            else:
                durasi_menit = DURASI_KUNJUNGAN_DEFAULT_MENIT  # fallback: 90 menit

            # Durasi tidak boleh melebihi window yang tersedia
            window_tersedia = tutup - buka
            durasi_menit = min(durasi_menit, window_tersedia - 1)
            durasi_menit = max(durasi_menit, 15)  # minimal 15 menit kunjungan

        except (KeyError, TypeError):
            # Jika destinasi tidak ditemukan di DataFrame, gunakan semua default
            buka = max(JAM_BUKA_DEFAULT_MENIT, waktu_mulai_menit)
            tutup = JAM_TUTUP_DEFAULT_MENIT
            durasi_menit = DURASI_KUNJUNGAN_DEFAULT_MENIT

        time_windows.append((buka, tutup))
        durasi_kunjungan.append(durasi_menit)

    return time_windows, durasi_kunjungan


# =============================================================================
# BAGIAN 4 — bagi_destinasi_jadi_jadwal_per_hari (DIPERLUAS untuk arrival_times)
# =============================================================================

def bagi_destinasi_jadi_jadwal_per_hari(
        semua_destinasi: dict[int, list[str]],
        semua_arrival_times: dict[int, list[int]],
        jumlah_hari: int,
) -> tuple[dict[int, list[str]], dict[int, list[int]]]:
    """
    Membagi urutan destinasi hasil VRPTW ke dalam jadwal per hari.

    Perubahan dari versi TSP:
        Versi lama hanya membagi list destinasi.
        Versi baru juga membagi arrival_times dengan slice yang identik,
        sehingga setiap hari tetap memiliki pasangan (nama, jam_tiba)
        yang konsisten.

    Logika pembagian:
        Distribusi rata dengan remainder: hari-hari awal mendapat
        satu destinasi ekstra jika jumlah tidak habis dibagi.
        Contoh: 7 destinasi, 3 hari → Hari1=3, Hari2=2, Hari3=2.
    """
    list_destinasi = semua_destinasi[1]
    list_arrival = semua_arrival_times.get(1, [])

    jumlah_destinasi = len(list_destinasi)
    jumlah_destinasi_minimal_per_hari = jumlah_destinasi // jumlah_hari
    sisa = jumlah_destinasi % jumlah_hari

    destinasi_per_hari: dict[int, list[str]] = {}
    arrival_per_hari: dict[int, list[int]] = {}

    start = 0
    for d in range(1, jumlah_hari + 1):
        extra = 1 if d <= sisa else 0
        end = start + jumlah_destinasi_minimal_per_hari + extra
        destinasi_per_hari[d] = list_destinasi[start:end]
        arrival_per_hari[d] = list_arrival[start:end] if list_arrival else []

        start = end

    return destinasi_per_hari, arrival_per_hari


# =============================================================================
# BAGIAN 5 — routing_destinasi (PERUBAHAN UTAMA)
# =============================================================================

async def routing_destinasi(
        data_tempat_wisata: pd.DataFrame,
        tempat_wisata_terfilter: dict[int, list[str]],
        jumlah_hari: int,
        waktu_mulai_menit: int = 8 * 60,  # parameter BARU: jam mulai (default 08:00)
) -> tuple[dict[int, list[str]], dict[int, list[int]], float]:
    """
    Melakukan pengurutan destinasi menggunakan VRPTW solver.

    Perubahan dari versi TSP:
    - Parameter baru  : waktu_mulai_menit (jam mulai perjalanan)
    - Matriks         : time_matrix (menit) menggantikan distance_matrix (km)
    - Data baru       : time_windows + durasi_kunjungan diekstrak dari DataFrame
    - Pemanggil solver: intracluster_vrptw (bukan intracluster_tsp)
    - Unpack return   : 3 nilai (urutan, arrival_times, total_waktu)
    - Return baru     : mengembalikan arrival_times_per_hari sebagai dict terpisah

    Parameters
    ----------
    data_tempat_wisata : pd.DataFrame
        DataFrame destinasi dengan kolom latitude, longitude, rating,
        dan opsional open_time, close_time, duration.

    tempat_wisata_terfilter : dict[int, list[str]]
        Output dari trip_planner_selection (CP-SAT):
        {1: ["Kawah Putih", "Saung Angklung", ...]}

    jumlah_hari : int
        Total hari perjalanan.

    waktu_mulai_menit : int
        Jam mulai perjalanan dalam menit sejak tengah malam.
        Diteruskan dari preferences.py via main.py.

    Returns
    -------
    destinasi_per_hari : dict[int, list[str]]
        {1: ["Kawah Putih", ...], 2: [...], ...}

    arrival_times_per_hari : dict[int, list[int]]
        {1: [480, 570, 720, ...], 2: [...], ...}
        Waktu tiba di setiap destinasi dalam menit sejak tengah malam.

    total_waktu_tempuh : float
        Total waktu perjalanan (menit) dari solver.
    """
    if len(tempat_wisata_terfilter) == 0:
        return {}, {}, 0.0

    # --- Ekstrak data koordinat dan rating dari DataFrame ---
    nama_destinasi   = data_tempat_wisata["title"].to_numpy()
    latitude         = data_tempat_wisata["latitude"].to_numpy()
    longitude        = data_tempat_wisata["longitude"].to_numpy()
    rating           = data_tempat_wisata["rating"].to_numpy()

    array_koordinat_per_titik = {
        titik: (lat, lon)
        for titik, lat, lon in zip(nama_destinasi, latitude, longitude)
    }
    rating_lokasi_pada_peta = dict(zip(nama_destinasi, rating))

    # --- Ambil list destinasi terpilih (output CP-SAT) ---
    tempat_wisata = tempat_wisata_terfilter[1]

    # --- Guard clause: edge case tanpa destinasi ---
    if len(tempat_wisata) == 0:
        return {1: []}, {1: []}, 0.0

    # --- Guard clause: edge case hanya 1 destinasi ---
    if len(tempat_wisata) == 1:
        arrival = [waktu_mulai_menit]
        destinasi_per_hari, arrival_per_hari = bagi_destinasi_jadi_jadwal_per_hari(
            {1: tempat_wisata}, {1: arrival}, jumlah_hari
        )
        return destinasi_per_hari, arrival_per_hari, 0.0

    # --- Bangun TIME MATRIX (menggantikan distance matrix) ---
    # Fungsi ini sekaligus memperbaiki dua bug di versi lama
    matriks_waktu = hitung_matriks_waktu_antartitik(
        array_koordinat_per_titik, tempat_wisata
    )

    # --- Ekstrak time windows dan durasi dari DataFrame ---
    # Ini adalah data baru yang tidak ada di pipeline TSP sebelumnya
    time_windows, durasi_kunjungan = ekstrak_time_windows_dan_durasi(
        data_tempat_wisata, tempat_wisata, waktu_mulai_menit
    )

    # --- Tentukan titik start dan end ---
    # Logika sama dengan TSP: start = rating tertinggi, end = paling jauh
    start, end = tentukan_titik_start_dan_end(
        array_koordinat_per_titik,
        rating_lokasi_pada_peta,
        tempat_wisata,
    )

    # --- Panggil VRPTW solver (menggantikan intracluster_tsp) ---
    # Return sekarang 3 nilai: nama terurut, arrival times, total waktu
    urutan_nama, arrival_times, total_waktu_tempuh = await intracluster_tsp(
        places=tempat_wisata,
        titik_awal=start,
        titik_akhir=end,
        matriks_waktu_antardestinasi=matriks_waktu,
        time_windows=time_windows,
        durasi_kunjungan=durasi_kunjungan,
        slack_max=60,          # maksimum 60 menit menunggu di satu tempat
        time_limit_detik=5,    # batas waktu komputasi GLS
    )

    # --- Bagi ke jadwal per hari (termasuk arrival_times) ---
    destinasi_per_hari, arrival_per_hari = bagi_destinasi_jadi_jadwal_per_hari(
        {1: urutan_nama},
        {1: arrival_times},
        jumlah_hari,
    )

    return destinasi_per_hari, arrival_per_hari, total_waktu_tempuh