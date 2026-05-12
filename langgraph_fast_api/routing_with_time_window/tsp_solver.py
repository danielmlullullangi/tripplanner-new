"""
Bagian pertama dari kode ini adalah import library-library yang dibutuhkan. Keseluruhan kode ini 
INTINYA adalah membuat suatu Solver Linear Programming dari Google ORTools yang cocok untuk
memecahkan permasalahan Traveling Salesperson Problem (TSP), tepatnya menentukan rute paling cost-effective.

Di sisi lain, TSP itu masalah optimasi yang kompleksitas komputasinya memiliki kategori NP-Hard. Artinya 
masalah pada kategori ini tidak dapat diverifikasi dengan cepat. Artinya, tidak ada algoritma yang saat ini diketahui 
yang ada untuk memecahkan masalah ini dengan efektif.

TSP sendiri mencari jawaban jika seorang salesperson yang harus mengunjungi sejumlah titik kota tepat 1 kali,
dan harus kembali ke kota asal, urutan kota seperti apa yang menghasilkan jarak atau cost (biaya) yang paling kecil.

Library yang diambil di bawah ini berasal dari pemecahan TSP dari Google OR-Tools > Routing > TSP
Link referensi: https://developers.google.com/optimization/routing/tsp#python_1
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp #library OR-Tools
import numpy as np
from utils import scaler
import asyncio

"""
Tujuan akhirnya adalah berhasilnya dibuat sebuah routing solver.
Untuk membuat routing solver, pertama didefinisikan dulu data-nya. Data ini
berisi jarak titik suatu destinasi terhadap semua titik destinasi lainnya, untuk semua destinasi.
Sehingga terbentuk n elemen jarak untuk semua m destinasi, dimana m=n sehingga List A[m][n].
Data inilah yang disebut sebagai distance_matrix/num_nodes pada function _solve_tsp_sync.

Selain itu, ada parameter start, end, depot, dan num_vehicles. Sebenarnya, ada 2 skema untuk RoutingIndexManager,
yaitu Single Depot dimana kendaraan berangkat dan berakhir di titik yang sama dan Custom Start & End dimana kendaraan
berangkat dan berakhir di titik yang berbeda. Pada solver di bawah, digunakan skema Custom Start & End karena tujuan dari 
kode ini digunakan untuk mencari destinasi yang berbeda (user tidak mungkin mengunjungi titik awal dan akhir yang sama).

RoutingIndexManager adalah function internal yang berperan sebagai "seorang manager" untuk bisa tahu struktur masalah routing 
agar nomor-nomor lokasi pada matriks distance_matrix bisa dipetakan ke bentuk integer internal-nya function 
-> intinya, masalah routing-nya didefinisikan jadi sesuai ketentuan solvernya.

Selanjutnya, data jarak antartitik pada distance_matrix. Tapi, karena masih dalam format list of list (raw), struktur ini
didefiniskan oleh RoutingIndexManager dan kemudian komponen jarak antartitiknya digunakan solver untuk mengakses distance
matrix kita. Nah, jembatan mengambil komponen jarak ini menggunakan function distance_callbanck.

Selanjutnya, distance_callback didaftarkan ke dalam sistem solver sehingga solver mengoutputkan integer ID (transit_callback_index).
Hal ini dilakukan karena solver OR-Tools ditulis dalam bahasa C++, tapi kita sebagai user menggunakan Python. Terdapat konsep FFI 
(Foreign Function Interface) sebagai mekanisme komunikasi 2 bahasa pemrograman yang berbeda. Nah, function ini close-over variabel 
data dan manager pada memori Python. Ketika function ini digunakan solver OR-Tools dalam C++, ia bisa mengakses variabel di luar dunia functionnya, 
meskipun sudah pindah bahasa pemrograman, which is variabel data dan manager.
"""

# PROPOSED
def solver_TSP_sync(
        matriks_waktu_antardestinasi: list[list[int]],  # ← ganti: time matrix, bukan distance
        time_windows: list[tuple[int, int]],               # ← BARU: [(open, close), ...] per destinasi
        durasi_kunjungan: list[int],                       # ← BARU: lama kunjungan per destinasi (menit)
        titik_awal: int | None = None,
        titik_akhir: int | None = None,
        slack_max: int = 60,                               # ← BARU: max waiting time (menit)
        time_limit_detik: int = 5
):

    jumlah_destinasi = len(matriks_waktu_antardestinasi)
    jumlah_kendaraan = 1

    """
    By default, permasalahan TSP (aslinya)--salesman mulai dari kota A, pergi ke SEMUA kota lain, lalu kembali ke kota A LAGI--itu disebut closed TSP karena
    rutenya tertutup (closed-loop) yang titik awal dan akhirnya sama.

    Bisa dimodifikasi permasalahan TSP untuk case ini, dimana titik awal dan titik akhirnya SELALU ditentukan. Pada case ini, conditional yang aktif 
    selalu pada 'else'-nya dan kondisi if dan elif semuanya tidak terpenuhi.
    Kondisi yang DIPAKAI saat ini adalah titik awalnya adalah destinasi dengan rating tertinggi dan titik akhirnya adalah destinasi dengan lokasi geografis (jarak) 
    yang paling jauh dari titik awalnya.
    """

    #TIDAK DIPAKAI
    if titik_awal is None and titik_akhir == None: # Create open tsp matrix
        new_matrix = [row + [0.0] for row in matriks_waktu_antardestinasi]
        new_matrix.append([0.0] * (jumlah_destinasi+1))

        manager = pywrapcp.RoutingIndexManager(
            jumlah_destinasi + 1,
            jumlah_kendaraan, # 1 vehicle
            [jumlah_destinasi], # start node, dummy node
            [jumlah_destinasi] # end node
        )
        matrix = new_matrix
        dummy_node = jumlah_destinasi

    #TIDAK DIPAKAI
    elif titik_awal is not None and titik_akhir is None:
        manager = pywrapcp.RoutingIndexManager(
            jumlah_destinasi,
            jumlah_kendaraan, # 1 vehicle
            titik_awal, # start node
        )
        matrix = matriks_waktu_antardestinasi
        dummy_node = None

    #Yang DIPAKAI saat ini hanya kondisi yang ini
    else:
        manager = pywrapcp.RoutingIndexManager(
            jumlah_destinasi,
            jumlah_kendaraan, # 1 vehicle
            [titik_awal], # start node
            [titik_akhir] # end node
        )
        matrix = matriks_waktu_antardestinasi
        dummy_node = None

    matrix = scaler(np.array(matrix), 1000)
    durasi_scaled = [int (d * 1000/ max(matriks_waktu_antardestinasi[0][0],1)) 
                     if max(max(row) for row in matriks_waktu_antardestinasi) > 0
                     else d
                     for d in durasi_kunjungan]
    
    faktor_skala = 1000/ max(max(row) for row in matriks_waktu_antardestinasi) if max (max(row) for row in matriks_waktu_antardestinasi) > 0 else 1
    durasi_scaled = [int(d * faktor_skala) for d in durasi_kunjungan]

    tw_scaled = [(int(buka*faktor_skala), int(tutup*faktor_skala)) for buka, tutup in time_windows]
    slack_scaled = int(slack_max*faktor_skala)

    kapasitas_waktu_scaled = max(tutup for _, tutup in tw_scaled) + slack_scaled

    routing = pywrapcp.RoutingModel(manager)


    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node] + durasi_scaled[from_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    """
    Selanjutnya, cost/biaya antara 2 lokasi yang berbeda ditambahkan pada SetArcCostEvaluatorofAllVehicle.
    Pada kode ini, cost antara 2 lokasi adalah transit_callback_index (yang ada di dalam function SetArc-nya).
    Sebenarnya, bisa dimodifikasi nih cost antara 2 titik selain mengandalkan jarak tempuhnya saja (transit_callback_index).
    Jadi, bisa di-ideate-kan suatu nilai cost baru.

    Selanjutnya, didefinisikan juga search space di mana search space ini berisi semua kemungkinan rute yang ada.
    Nah, alasannya didefinisikannya search space ini adalah untuk mengurangi banyaknya search space ideal.
    Misal, ada 13 titik lokasi, artinya untuk mencari 1 jalur ideal, dibutuhkan 12! (faktorial) kombinasi urutan destinasi, yaitu ~479jt kemungkinan ➝ sangat LAMA. →➜➞➝
    Untuk itu, di sini pemilihan kandidat pada search space dilakukan dengan lebih "cerdas", yaitu dengan menggunakan FirstSolutionStrategy.CHEAPEST_ARC.
    Sebenarnya FirstSolutionStrategy ini adalah fase pertama, tujuannya untuk menemukan solusi awal secara cepat dulu walau belum optimal. Fase pertama ini WAJIB pake FirstSolutionStrategy.
    Nah, di sini digunakan PATH_CHEAPEST_ARC,yaitu pemilihan node (titik/destinasi) berikutnya berdasarkan cost termurah.

    Sebenarnya, untuk kondisi banyak constraint yang harus dipenuhi, lebih cocok menggunakan PATH_MOST_CONSTRAINED_ARC,
    yaitu pemilihan node berikutnya yang paling terkonstrain. Analoginya seperti memprioritaskan orang yang paling sibuk karena time-window terbatas, bisa melayani kendaraan tertentu saja, dll.
    Untuk fase kedua, yaitu LocalSearchHeuristics (opsional bisa dipake or not). Sebenarnya, tidak harus LocalSearchHeuritstics. Ada variasi lain. Namun, paling umum digunakan pada fase 2 adalah LocalSearchHeuristics.
    """

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    NAMA_DIMENSI = "Time"
    routing.AddDimension(transit_callback_index,
                         slack_scaled,
                         kapasitas_waktu_scaled,
                         False,
                         NAMA_DIMENSI)
    time_dimension = routing.GetDimensionOrDie(NAMA_DIMENSI)

    for idx, (buka_scaled, tutup_scaled) in enumerate(tw_scaled):
        index = manager.NodeToIndex(idx)
        time_dimension.CumulVar(index).SetRange(buka_scaled, tutup_scaled)

    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(0)))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(0)))

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC
    )

    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    search_params.time_limit.seconds = 1

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return None, [], -1.0

    urutan_indeks = []
    arrival_time_scaled = []

    """
    Selanjutnya, proses looping dijalankan yang batasnya adalah titik destinasi terakhir (IsEnd(index)). Variabel index mendeskripsikan titik awal dimulainya rute (kendaraan ke-0).
    """
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        time_var = time_dimension.CumulVar(index)
        urutan_indeks.append(node)
        arrival_time_scaled.append(solution.Min(time_var))
        index = solution.Value(routing.NextVar(index))

    #Menambahkan titik terakhir. Kenapa perlu? Karena End node belum dimasukkan ke route karena loop sudah berhenti sebelum bisa append destinasinya.
    end_node = manager.IndexToNode(index)
    time_var_end = time_dimension.CumulVal(index)
    urutan_indeks.append(end_node)
    arrival_time_scaled.append(solution.Min(time_var_end))

    arrival_time_menit = [int(t/faktor_skala) for t in arrival_time_scaled]

    total_waktu_tempuh = solution.ObjectiveValue() / 1000

    return urutan_indeks, arrival_time_menit, total_waktu_tempuh

async def solver_TSP_async(
        matriks_waktu_antardestinasi: list[list[float]],
        time_windows: list[tuple[int, int]],
        durasi_kunjungan: list[int],
        titik_awal: int | None = None,
        titik_akhir: int | None = None,
        slack_max: int = 60,
        time_limit_detik: int = 5,
) -> tuple[list[int] | None, list[int], float]:
    return await asyncio.to_thread(solver_TSP_sync, 
                                   matriks_waktu_antardestinasi,
                                   time_windows,
                                   durasi_kunjungan,
                                   titik_awal, 
                                   titik_akhir,
                                   slack_max,
                                   time_limit_detik)

#Function FINAL yang dioper ke routing_core.py
async def intracluster_tsp(
        places: list[str],
        titik_awal: int | None,
        titik_akhir: int | None,
        matriks_waktu_antardestinasi: list[list[float]],
        time_windows: list[tuple[int, int]],
        durasi_kunjungan: list[int],
        slack_max: int = 60,
        time_limit_detik: int = 5,
) -> tuple[list[str], list[int], float]:
    urutan_indeks, arrival_time, total_waktu_tempuh = await solver_TSP_async(matriks_waktu_antardestinasi, 
                                                                             time_windows,
                                                                             durasi_kunjungan,
                                                                             titik_awal, 
                                                                             titik_akhir,
                                                                             slack_max,
                                                                             time_limit_detik)
    if urutan_indeks is None:
        arrival_time_fallback = [time_windows[i][0] for i in range(len(places))]
        return places, arrival_time_fallback, 0.0

    urutan_nama = [places[i] for i in urutan_indeks]
    
    return urutan_nama, arrival_time, total_waktu_tempuh