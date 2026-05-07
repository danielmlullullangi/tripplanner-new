from haversine import haversine
import pandas as pd
from routing.tsp_solver import intracluster_tsp


"""

"""
def tentukan_titik_start_dan_end(
        koordinat_lokasi_pada_peta: dict[str, tuple[float, float]],
        scores_map: dict[str, int],
        ordered: list[str],
) -> tuple[int, int]:
    """
    Menentukan start dan end point berdasarkan tempat dengan rating tertinggi atau terdekat
    """
    jumlah_tempat_dalam_cluster = len(ordered)
    if jumlah_tempat_dalam_cluster == 0:
        return -1, -1 # Menandakan tidak ada tempat dalam cluster
    if jumlah_tempat_dalam_cluster == 1:
        return 0, 0   # Hanya 1 tempat, titik start dan end di tempat yang sama

    indeks_titik_pertama, indeks_titik_terakhir = -1, -1

    # Menentukan start_idx
    # Cluster (hari) pertama
    indeks_titik_pertama = ordered.index(max(ordered, key=lambda p: scores_map.get(p, 0))) # Heuristic: mengambil tempat dengan rating tertinggi

    # Mencari end_idx dengan titik terjauh dengan start_idx
    start_coord = koordinat_lokasi_pada_peta[ordered[indeks_titik_pertama]]
    jarak = -1

    for tempat in range(jumlah_tempat_dalam_cluster):
        if tempat == indeks_titik_pertama:
            continue

        d = haversine(start_coord, koordinat_lokasi_pada_peta[ordered[tempat]])
        if d > jarak:
            jarak = d
            indeks_titik_terakhir = tempat

    return indeks_titik_pertama, indeks_titik_terakhir

def get_distance_matrix(
        koordinat_lokasi_pada_peta: dict[str, tuple[float, float]],
        places: list[str]
) -> list[list[float]]:
    """
    Menghitung distance matrix berdasarkan
    tempat yang sudah diketahui
    """
    n = len(places)
    coords = [koordinat_lokasi_pada_peta[p] for p in places]

    matrix = [[0.0] * n for _ in range(n)]

    # hitung hanya segitiga atas (karena simetris)
    for i in range(n):
        c1 = coords[i]
        for j in range(i + 1, n):
            d = haversine(c1, coords[j])
            matrix[i][j] = d
            matrix[j][i] = d

    return matrix

def itinerary(
        P: dict[int, list[str]],
        D: int
) -> dict[int, list[str]]:

    result_all = P[1]
    n = len(result_all)

    n_per_day = n // D
    remainder = n % D

    P_processed = {}
    start = 0
    
    for d in range(1, D+1):
        extra = 1 if d <= remainder else 0
        end = start + n_per_day + extra
        P_processed[d] = result_all[start:end]
        start = end

    return P_processed

async def destination_routing(
        query: pd.DataFrame,
        clusters: dict[int, list[str]],
        D: int
) -> dict[int, list[str]]:
    """
    Melakukan pengurutan destinasi intercluster, lalu intracluster
    """
    if len(clusters) == 0:
        return {}, 0.0

    # ekstraks koordinat dan score (rating)
    titles = query["title"].to_numpy()
    lat = query["latitude"].to_numpy()
    lon = query["longitude"].to_numpy()
    rating = query["rating"].to_numpy()

    coordinates_map = {
        t: (la, lo)
        for t, la, lo in zip(titles, lat, lon)
    }

    scores_map = dict(zip(titles, rating))

    # Membuat distance matrix untuk setiap cluster
    dist_mat = {1: get_distance_matrix(coordinates_map, clusters[1])}

    # Optimisasi urutan destinasi di setiap cluster (intracluster)
    P = {1: []}
    places = clusters[1]

    if len(places) == 0:
        P[1] = []
        return P, 0.0

    elif len(places) == 1:
        P[1] = places
        return P, 0.0

    start, end = tentukan_titik_start_dan_end(
            coordinates_map,
            scores_map,
            places
        )
    
    P[1], dist = await intracluster_tsp(
            places,
            start,
            end,
            dist_mat[1],
        )

    P = itinerary(P, D)

    return P, dist