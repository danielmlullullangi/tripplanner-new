from utils import scaler
import pandas as pd
from haversine import haversine_vector, Unit
from ortools.sat.python import cp_model
import asyncio

# query difilter berdasarkan radius -> CP-SAT

def query_masking(query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(query) == 0:
        return query, query

    th = 40
    coords = query[["latitude", "longitude"]].values
    center = coords.mean(axis=0)

    d = haversine_vector(center, coords, Unit.KILOMETERS, comb=True)
    mask = d <= th

    return query.loc[mask], query.loc[~mask]

def _solve_and_collect(
    N: int,
    x_var: dict[tuple[int, int], cp_model.IntVar],
    places: list[str],
    cost: list[int],
    model: cp_model.CpModel,
):

    solver = cp_model.CpSolver()
    # solver.parameters.num_search_workers = 10
    status = solver.Solve(model)
    result = {1: []}

    total_cost = 0
    
    if status == cp_model.OPTIMAL:
        for i in range(N):
            if solver.value(x_var[i]) == 1:
                result[1].append(places[i])
                total_cost += cost[i]

        return result, True, total_cost

    return result, False, total_cost

def _create_solver(data: pd.DataFrame, data_original, D, budget, time_limit, alternative: bool = False):        
        rating_real = data["rating_total"].values
        cost_real = data["price_mean"].values
        duration_real = data["duration"].values
        cost_original = data_original["price_mean"].values
        
        places = data["title"].values
        rating = scaler(rating_real, 100)
        cost = scaler(cost_real, 100)
        duration = scaler(duration_real, 100)
        N = len(places)

        model = cp_model.CpModel()

        # Buat variabel
        ## x[i][d] = 1 jika tempat i dikunjungi hari d
        x = {
            i: model.new_int_var(0, 1, f"x_{i}")
            for i in range(N)
        }

        # Objective function: maximize rating
        model.maximize(
            sum(rating[i] * x[i] for i in range(N))
        )

        if not alternative:
            # Constraint 1: minimal D*1 tempat
            model.add(sum(x[i] for i in range(N)) >= D*1)

        # Constraint 2: tempat maksimal dikunjungi 1 kali
        for i in range(N):
            model.add(x[i] <= 1)

        # Constraint 3: total biaya
        model.add(
            sum(cost[i] * x[i] for i in range(N)) <= budget * 100
        )

        # Constraint 4: batas waktu wisata
        model.add(
            sum(duration[i] * x[i] for i in range(N)) <= sum(time_limit) * 100
        )

        if not alternative:
            # Constraint 5: maksimal D*1 tempat yang harganya 0
            model.add(
                sum(x[i] for i in range(N) if cost[i] == 0) <= D*1
            )

        # Constraint 6: maksimal D*10 tempat
        model.add(
            sum(x[i] for i in range(N)) <= D*10
        )

        return places, cost_original, duration_real, N, x, model

def _run_solver_sync(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    D: int,
    budget: int | float,
    time_limit: list[float],
    alternative: bool = False,
):
    places, cost, duration, N, x, model = _create_solver(
        data, data_original, D, budget, time_limit, alternative
    )
    return _solve_and_collect(N, x, places, cost, model)

async def _run_solver(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    D: int,
    budget: int | float,
    time_limit: list[float],
    alternative: bool = False,
):
    return await asyncio.to_thread(
        _run_solver_sync, data, data_original, D, budget, time_limit, alternative
    )

# Linear Optimization (SAT) (sama aja, tapi nilai constraint nya bisa desimal)
async def trip_planner_selection(
    query: pd.DataFrame, # enhanced
    query_original: pd.DataFrame,
    D: int,
    budget: int | float,
    time_limit: list[float],
    pretty_print: bool = False
) -> dict[int, list[str]]:

    def flatten(res):
        return res[1]

    query_masked, query_out = query_masking(query)
    query_masked_ori, query_out_ori = query_masking(query_original)

    result, success, total_cost = await _run_solver(
        query_masked, query_masked_ori, D, budget, time_limit
    )
    res = flatten(result)

    if not res and len(query_masked) > 0 and not success:
        N = len(query_masked)

        if N <= D:
            # Ambil tempat seadanya
            total_cost = 0
            result = {1: []}
            print("Masih ada tempat di dalam radius, (N <= D). Mengambil tempat seadanya.")
            cost = query_masked["price_max"].values
            places = query_masked["title"].values
            for i in range(N):
                result[1].append(places[i])
                total_cost += cost[i]
        else: # N > D
            # Gunakan constraint alternatif
            print("Masih ada tempat di dalam radius, (N > D). Menggunakan constraint alternatif.")
            result, success, total_cost = await _run_solver(
                query_masked, query_masked_ori, D, budget, time_limit, alternative=True
            )

    res = flatten(result)

    if not res and len(query_out) > 0:
        # Ambil tempat yang gabung dengan yang ada di luar radius
        N = len(query)

        if N <= D:
            # Ambil tempat seadanya
            total_cost = 0
            result = {1: []}
            print("Tidak ada solusi di dalam radius, (N <= D).\nMenggunakan data gabungan.\nMengambil tempat seadanya.")
            cost = query["price_max"].values
            places = query["title"].values
            for i in range(N):
                result[1].append(places[i])
                total_cost += cost[i]
        else: # N > D
            # Gunakan constraint utama, dengan data query gabungan
            print("Tidak ada solusi di dalam radius, (N > D). Menggunakan data gabungan.")
            result, success, total_cost = await _run_solver(
                query, query_original, D, budget, time_limit
            )

            # Apabila masih kosong dan gagal, gunakan constraint alternatif, dengan data query gabungan
            if not flatten(result) and not success:
                print("Tidak ada solusi di dalam radius, (N > D). Menggunakan data gabungan dan constraint alternatif.")
                result, success, total_cost = await _run_solver(
                    query, query_original, D, budget, time_limit, alternative=True
                )
    
    return result, int(total_cost)