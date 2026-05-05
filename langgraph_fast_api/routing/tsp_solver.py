"""
Import library-library yang dibutuhkan. Solver Linear Programming dari Google ORTools di sini digunakan 
untuk memecahkan permasalahan Traveling Salesperson Problem. 
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import numpy as np
from utils import scaler
import asyncio

def _solve_tsp_sync(
        distance_matrix: list[list[float]],
        start: int | None = None,
        end: int | None = None,
):
    """
    Penyelesaian urutan kunjungan pada permasalahan TSP.
    """
    #
    n = len(distance_matrix)

    if start is None and end == None: # Create open tsp matrix
        new_matrix = [row + [0.0] for row in distance_matrix]
        new_matrix.append([0.0] * (n+1))

        manager = pywrapcp.RoutingIndexManager(
            n + 1,
            1, # 1 vehicle
            [n], # start node
            [n] # end node
        )
        matrix = new_matrix
        dummy_node = n

    elif start is not None and end is None:
        manager = pywrapcp.RoutingIndexManager(
            n,
            1, # 1 vehicle
            start, # start node
        )
        matrix = distance_matrix
        dummy_node = None

    else:
        manager = pywrapcp.RoutingIndexManager(
            n,
            1, # 1 vehicle
            [start], # start node
            [end] # end node
        )
        matrix = distance_matrix
        dummy_node = None

    matrix = scaler(np.array(matrix), 1000)

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # search_params.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # )
    # search_params.time_limit.seconds = 1

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return None, -1

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != dummy_node:
            route.append(node)
        index = solution.Value(routing.NextVar(index))

    end_node = manager.IndexToNode(index)
    if end_node != dummy_node:
        route.append(end_node)

    return route, solution.ObjectiveValue()/1000

async def solve_tsp(
        distance_matrix: list[list[float]],
        start: int | None = None,
        end: int | None = None,
):

    return await asyncio.to_thread(_solve_tsp_sync, distance_matrix, start, end)

async def intracluster_tsp(
        places: list[str],
        start: int | None,
        end: int | None,
        distance_m: list[list[float]],
):

    solution, total_dist = await solve_tsp(distance_m, start, end)
    solution = [places[i] for i in solution]

    return solution, total_dist




