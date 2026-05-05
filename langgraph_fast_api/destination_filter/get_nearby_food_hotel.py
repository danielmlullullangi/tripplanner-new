import pandas as pd
from haversine import haversine
import asyncio

def get_all_accommodation_data(
        dt: pd.DataFrame,
        category: str,  # "food" or "hotel"
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:

    if category == "food":
        mask = (
            (dt["category"] == "Food & Drink") &
            (dt["poi"] != "event")
        )
    elif category == "hotel":
        mask = (
            (dt["category"] == "Hotels & Accomodations") &
            (dt["poi"] != "event")
        )
    else:
        raise ValueError("Unknown category")

    req_dat = dt.loc[mask]
    req_map = (
        req_dat
        .set_index('title')[['latitude', 'longitude']]
        .apply(tuple, axis=1)
        .to_dict()
    )

    return req_dat, req_map

def _process_day(
        d: int,
        places_for_day: list,
        food_map: dict,
        hotel_map: dict,
        food_dat_indexed: pd.DataFrame,
        hotel_dat_indexed: pd.DataFrame,
        query_indexed: pd.DataFrame,
        dist: float,
        N_top: int,
) -> tuple:
    
    food_candidates_for_day = {}
    hotel_candidates_for_day = {}

    for place in places_for_day:
        # Get coordinates for the current place
        coord0 = tuple(
                query_indexed
                .loc[place][["latitude", "longitude"]]
                .values
        )

        def get_all_accommodation(
                req_map: dict[str, tuple[float, float]],
                candidates: dict[str, tuple[int, float]],
                dat_indexed: pd.DataFrame
        ) -> dict[str, tuple[float, float]]:

            for p_req, coord_req in req_map.items():
                distance = haversine(coord0, coord_req)
                if distance <= dist and p_req not in candidates:
                    rating_total = dat_indexed.loc[p_req, "rating_total"]
                    if isinstance(rating_total, pd.DataFrame):
                        rating_total = rating_total.iloc[0]
                    rating_total = float(rating_total)
                    candidates[p_req] = (rating_total, distance)

            return candidates

        # Collect food candidates for the day
        food_candidates_for_day = get_all_accommodation(
            food_map,
            food_candidates_for_day,
            food_dat_indexed
        )

        # Collect hotel candidates for the day
        hotel_candidates_for_day = get_all_accommodation(
            hotel_map,
            hotel_candidates_for_day,
            hotel_dat_indexed
        )

    def top_n(
            candidates: dict[str, tuple[int, float]]
    ) -> dict[int, list[str]]:

        if not candidates:
            return []
        
        sorted_items = sorted(
                candidates.items(),
                key=lambda item: (-item[1][0], item[1][1])
        )
        return [item[0] for item in sorted_items[:N_top]]

    return d, top_n(food_candidates_for_day), top_n(hotel_candidates_for_day)

# Get 5 top (rating, distance) food and hotel
async def get_top_food_hotel(
        dat: pd.DataFrame,
        query: pd.DataFrame,
        result: dict[int, list[str]],
        N_days: int,
        N_top: int = 5
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:

    food_dat, food_map = get_all_accommodation_data(dat, "food")
    hotel_dat, hotel_map = get_all_accommodation_data(dat, "hotel")

    food = {d: [] for d in range(1, N_days+1)} # food
    hotel = {d: [] for d in range(1, N_days+1)} # hotel
    dist = 5 # km

    food_dat_indexed = food_dat.set_index('title')
    hotel_dat_indexed = hotel_dat.set_index('title')
    query_indexed = query.set_index('title')

    tasks = [
        asyncio.to_thread(
            _process_day,
            d,
            result[d],
            food_map,
            hotel_map,
            food_dat_indexed,
            hotel_dat_indexed,
            query_indexed,
            dist,
            N_top,
        )
        for d in range(1, N_days + 1)
    ]

    # Semua hari jalan bersamaan
    results = await asyncio.gather(*tasks)

    food = {d: [] for d in range(1, N_days + 1)}
    hotel = {d: [] for d in range(1, N_days + 1)}

    for d, food_top, hotel_top in results:
        food[d] = food_top
        hotel[d] = hotel_top
        
    return food, hotel