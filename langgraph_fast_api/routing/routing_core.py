from haversine import haversine
import pandas as pd
from routing.tsp_solver import intracluster_tsp

def tentukan_titik_start_dan_end(
        koordinat_lokasi_pada_peta: dict[str, tuple[float, float]],
        rating_lokasi_pada_peta: dict[str, int],
        titik_lokasi_terfilter: list[str],
) -> tuple[int, int]:
    """
    Menentukan start dan end point berdasarkan tempat dengan rating tertinggi atau terdekat
    """
    jumlah_tempat_dalam_cluster = len(titik_lokasi_terfilter)
    if jumlah_tempat_dalam_cluster == 0:
        return -1, -1 # Menandakan tidak ada tempat dalam cluster
    if jumlah_tempat_dalam_cluster == 1:
        return 0, 0   # Hanya 1 tempat, titik start dan end di tempat yang sama

    indeks_titik_pertama, indeks_titik_terakhir = -1, -1

    # Menentukan start_idx
    # Cluster (hari) pertama
    indeks_titik_pertama = titik_lokasi_terfilter.index(max(titik_lokasi_terfilter, key=lambda p: rating_lokasi_pada_peta.get(p, 0))) # Heuristic: mengambil tempat dengan rating tertinggi

    # Mencari end_idx dengan titik terjauh dengan start_idx
    start_coord = koordinat_lokasi_pada_peta[titik_lokasi_terfilter[indeks_titik_pertama]]
    jarak = -1

    for tempat in range(jumlah_tempat_dalam_cluster):
        if tempat == indeks_titik_pertama:
            continue

        jarak_haversine = haversine(start_coord, koordinat_lokasi_pada_peta[titik_lokasi_terfilter[tempat]])
        if jarak_haversine > jarak:
            jarak = jarak_haversine
            indeks_titik_terakhir = tempat
    return indeks_titik_pertama, indeks_titik_terakhir

def hitung_matriks_jarak_antartitik(
        koordinat_lokasi_pada_peta: dict[str, tuple[float, float]],
        daftar_nama_tempat: list[str]
) -> list[list[float]]:
    """
    Menghitung distance matrix berdasarkan
    tempat yang sudah diketahui
    """
    jumlah_tempat = len(daftar_nama_tempat)
    koordinat_terurut_bds_lokasi = [koordinat_lokasi_pada_peta[p] for p in places]

    matrix = [[0.0] * jumlah_tempat for _ in range(jumlah_tempat)]

    # hitung hanya segitiga atas (karena simetris)
    for counter_baris in range(jumlah_tempat):
        koordinat_tempat = koordinat_terurut_bds_lokasi[counter_baris]
        for counter_kolom in range(counter_baris + 1, jumlah_tempat):
            d = haversine(koordinat_tempat, koordinat_terurut_bds_lokasi[counter_kolom])
            matrix[counter_baris][counter_kolom] = d
            matrix[counter_baris][counter_kolom] = d
    return matrix

def bagi_destinasi_jadi_jadwal_per_hari(
        semua_destinasi: dict[int, list[str]],
        jumlah_hari: int
) -> dict[int, list[str]]:

    list_destinasi = semua_destinasi[1]
    jumlah_destinasi = len(list_destinasi)

    jumlah_destinasi_minimal_per_hari = jumlah_destinasi // jumlah_hari
    sisa_bagi = jumlah_destinasi % jumlah_hari

    destinasi_dibagi_per_hari = {}
    start = 0
    
    for d in range(1, jumlah_hari+1):
        extra = 1 if d <= sisa_bagi else 0
        end = start + jumlah_destinasi_minimal_per_hari + extra
        destinasi_dibagi_per_hari[d] = list_destinasi[start:end]
        start = end

    return destinasi_dibagi_per_hari

async def routing_destinasi(
        data_tempat_wisata: pd.DataFrame,
        tempat_wisata_terfilter: dict[int, list[str]],
        jumlah_hari: int
) -> dict[int, list[str]]:
    """
    Melakukan pengurutan destinasi intercluster, lalu intracluster
    """
    if len(tempat_wisata_terfilter) == 0:
        return {}, 0.0

    # ekstraks koordinat dan score (rating)
    nama_destinasi = data_tempat_wisata["title"].to_numpy()
    latitude = data_tempat_wisata["latitude"].to_numpy()
    longitude = data_tempat_wisata["longitude"].to_numpy()
    rating = data_tempat_wisata["rating"].to_numpy()

    array_koordinat_per_titik = {
        titik: (lat, lon)
        for titik, lat, lon in zip(nama_destinasi, latitude, longitude)
    }

    rating_lokasi_pada_peta = dict(zip(nama_destinasi, rating))

    # Membuat distance matrix untuk setiap cluster
    matriks_jarak = {1: hitung_matriks_jarak_antartitik(array_koordinat_per_titik, tempat_wisata_terfilter[1])}

    # Optimisasi urutan destinasi di setiap cluster (intracluster)
    titik_hasil_TSP_dibagi_per_hari = {1: []}
    tempat_wisata = tempat_wisata_terfilter[1]

    if len(tempat_wisata) == 0:
        titik_hasil_TSP_dibagi_per_hari[1] = []
        return titik_hasil_TSP_dibagi_per_hari, 0.0

    elif len(tempat_wisata) == 1:
        titik_hasil_TSP_dibagi_per_hari[1] = tempat_wisata
        return titik_hasil_TSP_dibagi_per_hari, 0.0

    start, end = tentukan_titik_start_dan_end(
            array_koordinat_per_titik,
            rating_lokasi_pada_peta,
            tempat_wisata
        )
    
    titik_hasil_TSP_dibagi_per_hari[1], total_jarak_perjalanan = await intracluster_tsp(
            tempat_wisata,
            start,
            end,
            matriks_jarak[1],
        )
    titik_hasil_TSP_dibagi_per_hari = bagi_destinasi_jadi_jadwal_per_hari(titik_hasil_TSP_dibagi_per_hari, jumlah_hari)

    return titik_hasil_TSP_dibagi_per_hari, total_jarak_perjalanan