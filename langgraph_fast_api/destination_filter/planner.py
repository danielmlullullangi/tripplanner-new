from utils import scaler
import pandas as pd
from haversine import haversine_vector, Unit
from ortools.sat.python import cp_model
import asyncio

#Query difilter berdasarkan radius -> CP-SAT
#Mengambil tempat-tempat yang ada di dalam radius saja
def filter_tempat_bds_radius(data_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data_input) == 0:
        return data_input, data_input

    threshold_radius = 40 #dalam kilometer
    array_koordinat_tempat_wisata = data_input[["latitude", "longitude"]].values
    koordinat_rerata_tempat_wisata = array_koordinat_tempat_wisata.mean(axis=0)

    jarak_titik_tengah_ke_setiap_tempat_wisata = haversine_vector(koordinat_rerata_tempat_wisata, array_koordinat_tempat_wisata, Unit.KILOMETERS, comb=True)
    apakah_titik_kurang_dari_radius = jarak_titik_tengah_ke_setiap_tempat_wisata <= threshold_radius

    return data_input.loc[apakah_titik_kurang_dari_radius], data_input.loc[~apakah_titik_kurang_dari_radius]


"""

"""

def solver_CP_SAT(
    jumlah_tempat_wisata: int,
    bool_pemilihan_tempat_oleh_solver: dict[tuple[int, int], cp_model.IntVar],
    daftar_nama_tempat: list[str],
    harga_tiap_tempat: list[int],
    model: cp_model.CpModel,
):

    solver = cp_model.CpSolver()
    # solver.parameters.num_search_workers = 10
    status_hasil_solve = solver.Solve(model)
    tempat_yg_terpilih = {1: []}

    jumlah_biaya = 0
    
    if status_hasil_solve == cp_model.OPTIMAL:
        for i in range(jumlah_tempat_wisata):
            if solver.value(bool_pemilihan_tempat_oleh_solver[i]) == 1:
                tempat_yg_terpilih[1].append(daftar_nama_tempat[i])
                jumlah_biaya += harga_tiap_tempat[i]

        return tempat_yg_terpilih, True, jumlah_biaya

    return tempat_yg_terpilih, False, jumlah_biaya

def model_terkonfigurasi(
        data_input_modified: pd.DataFrame, 
        data_input_original, 
        jumlah_hari, 
        budget, 
        time_limit, 
        alternative: bool = False
        ):        
        rating = data_input_modified["rating_total"].values
        harga = data_input_modified["price_mean"].values
        durasi_modified = data_input_modified["duration"].values
        harga_rerata_original = data_input_original["price_mean"].values
        
        nama_tempat_wisata = data_input_modified["title"].values
        rating = scaler(rating, 100)
        harga = scaler(harga, 100)
        duration = scaler(durasi_modified, 100)
        jumlah_tempat_wisata = len(nama_tempat_wisata)

        model = cp_model.CpModel()

        # Buat variabel
        ## x[i][d] = 1 jika tempat i dikunjungi hari d
        bool_terpilih_tidak = {
            i: model.new_int_var(0, 1, f"x_{i}")
            for i in range(jumlah_tempat_wisata)}

        # Objective function: maximize rating
        model.maximize(
            sum(rating[i] * bool_terpilih_tidak[i] for i in range(jumlah_tempat_wisata)))

        if not alternative:
            # Constraint 1: minimal D*1 tempat
            model.add(sum(bool_terpilih_tidak[i] for i in range(jumlah_tempat_wisata)) >= jumlah_hari*1)

        # Constraint 2: tempat maksimal dikunjungi 1 kali
        for i in range(jumlah_tempat_wisata):
            model.add(bool_terpilih_tidak[i] <= 1)

        # Constraint 3: total biaya
        model.add(
            sum(harga[i] * bool_terpilih_tidak[i] for i in range(jumlah_tempat_wisata)) <= budget * 100)

        # Constraint 4: batas waktu wisata
        model.add(
            sum(duration[i] * bool_terpilih_tidak[i] for i in range(jumlah_tempat_wisata)) <= sum(time_limit) * 100)

        if not alternative:
            # Constraint 5: maksimal D*1 tempat yang harganya 0
            model.add(
                sum(bool_terpilih_tidak[i] for i in range(jumlah_tempat_wisata) if harga[i] == 0) <= jumlah_hari*1)

        # Constraint 6: maksimal D*10 tempat
        model.add(
            sum(bool_terpilih_tidak[i] for i in range(jumlah_tempat_wisata)) <= jumlah_hari*10
        )
        return nama_tempat_wisata, harga_rerata_original, durasi_modified, jumlah_tempat_wisata, bool_terpilih_tidak, model

def jalankan_solver(
    data_input_modified, 
    data_input_original, 
    jumlah_hari, 
    budget, 
    time_limit, 
    alternative: bool = False
):
    daftar_nama_tempat, harga_tiap_tempat, _, jumlah_tempat_wisata, bool_pemilihan_tempat_oleh_solver, model = model_terkonfigurasi(
        data_input_modified, 
        data_input_original, 
        jumlah_hari, 
        budget, 
        time_limit, 
        alternative
    )
    return solver_CP_SAT(jumlah_tempat_wisata, bool_pemilihan_tempat_oleh_solver, daftar_nama_tempat, harga_tiap_tempat, model)

async def jalankan_solver_async(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    jumlah_hari: int,
    budget: int | float,
    time_limit: list[float],
    alternative: bool = False,
):
    return await asyncio.to_thread(
        jalankan_solver, data, data_original, jumlah_hari, budget, time_limit, alternative
    )

# Linear Optimization (SAT) (sama aja, tapi nilai constraint nya bisa desimal)
async def pemilihan_titik_wisata(
    data_input_modified: pd.DataFrame, # enhanced
    data_input_original: pd.DataFrame,
    jumlah_hari: int,
    budget: int | float,
    time_limit: list[float],
    pretty_print: bool = False
) -> dict[int, list[str]]:

    def flatten(res):
        return res[1]

    tempat_wisata_dalam_radius, tempat_wisata_luar_radius = filter_tempat_bds_radius(data_input_modified)
    tempat_wisata_dalam_radius_original, _ = filter_tempat_bds_radius(data_input_original)

    tempat_terpilih, berhasil_tidak, total_biaya = await jalankan_solver_async(
        tempat_wisata_dalam_radius, 
        tempat_wisata_dalam_radius_original, 
        jumlah_hari, 
        budget, 
        time_limit
    )
    res = flatten(tempat_terpilih)

    if not res and len(tempat_wisata_dalam_radius) > 0 and not berhasil_tidak:
        jumlah_tempat_tersedia = len(tempat_wisata_dalam_radius)

        if jumlah_tempat_tersedia <= jumlah_hari:
            # Ambil tempat seadanya
            total_biaya = 0
            tempat_terpilih = {1: []}
            print("Masih ada tempat di dalam radius, (N <= D). Mengambil tempat seadanya.")
            cost = tempat_wisata_dalam_radius["price_max"].values
            places = tempat_wisata_dalam_radius["title"].values
            for i in range(jumlah_tempat_tersedia):
                tempat_terpilih[1].append(places[i])
                total_biaya += cost[i]
        else: # N > D
            # Gunakan constraint alternatif
            print("Masih ada tempat di dalam radius, (N > D). Menggunakan constraint alternatif.")
            tempat_terpilih, berhasil_tidak, total_biaya = await jalankan_solver_async(
                tempat_wisata_dalam_radius, 
                tempat_wisata_dalam_radius_original, 
                jumlah_hari, 
                budget, 
                time_limit, 
                alternative=True
            )

    res = flatten(tempat_terpilih)

    if not res and len(tempat_wisata_luar_radius) > 0:
        # Ambil tempat yang gabung dengan yang ada di luar radius
        jumlah_tempat_tersedia = len(data_input_modified)

        if jumlah_tempat_tersedia <= jumlah_hari:
            # Ambil tempat seadanya
            total_biaya = 0
            tempat_terpilih = {1: []}
            print("Tidak ada solusi di dalam radius, (N <= D).\nMenggunakan data gabungan.\nMengambil tempat seadanya.")
            cost = data_input_modified["price_max"].values
            places = data_input_modified["title"].values
            for i in range(jumlah_tempat_tersedia):
                tempat_terpilih[1].append(places[i])
                total_biaya += cost[i]
        
        else: # N > D
            # Gunakan constraint utama, dengan data query gabungan
            print("Tidak ada solusi di dalam radius, (N > D). Menggunakan data gabungan.")
            tempat_terpilih, berhasil_tidak, total_biaya = await jalankan_solver_async(
                data_input_modified, 
                data_input_original, 
                jumlah_hari, 
                budget, 
                time_limit
            )

            # Apabila masih kosong dan gagal, gunakan constraint alternatif, dengan data query gabungan
            if not flatten(tempat_terpilih) and not berhasil_tidak:
                print("Tidak ada solusi di dalam radius, (N > D). Menggunakan data gabungan dan constraint alternatif.")
                tempat_terpilih, berhasil_tidak, total_biaya = await jalankan_solver_async(
                    data_input_modified, 
                    data_input_original, 
                    jumlah_hari, 
                    budget, 
                    time_limit, 
                    alternative=True
                )
    
    return tempat_terpilih, int(total_biaya)