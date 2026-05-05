from typing_extensions import TypedDict, Any
import pandas as pd
from destination_filter.preferences import get_preferences
from destination_filter.planner import trip_planner_selection
from routing.routing_core import destination_routing
from destination_filter.get_nearby_food_hotel import get_top_food_hotel
from utils import generate_metadata

async def generate_itinerary(data_input, data_enhanced, days, whom, styles, budget, destination):
    # Ubah data dari supabase (list of dict) ke dataframe
    dat = pd.DataFrame(data_input)
    dat_enhanced = pd.DataFrame(data_enhanced)
    
    query_enhanced = dat_enhanced.loc[(dat["category"] != "Hotels & Accomodations") &
                    (dat["category"] != "Food & Drink") &
                    (dat["poi"] != "event")]

    query = dat.loc[(dat["category"] != "Hotels & Accomodations") &
                    (dat["category"] != "Food & Drink") &
                    (dat["poi"] != "event")]

    query_accom = dat.loc[((dat["category"] == "Hotels & Accomodations") |
                          (dat["category"] == "Food & Drink")) &
                          (dat["poi"] != "event")]

    # Get preferences
    print("HAHA1")
    time_limit, query = await get_preferences(query, days, whom, styles)
    query_enhanced = query_enhanced.loc[query.index]
    # print("HAHA2")
    # Search destination
    result, total_destination_cost = await trip_planner_selection(query_enhanced, query, days, budget, time_limit, False)
    # print("HAHA3")
    # Destination routing
    P, total_dist = await destination_routing(query, result, days)
    # print("HAHA4")
    # Find nearest accommodation
    food, hotel = await get_top_food_hotel(query_accom, query, P, days, 5)
    # print("HAHA5")

    P = generate_metadata(P, query)
    food = generate_metadata(food, query_accom)
    hotel = generate_metadata(hotel, query_accom)
    # print("HAHA6")

    def mean_price(
            res: dict[int, dict[str, dict[str, Any]]]
    ) -> int:
        total = count = 0
        for day in res.values():
            for place in day.values():
                total += place["price_mean"]
                count += 1

        res_mean = int(total / count) if count else 0
        return res_mean

    # print("HAHA6")
    food_mean = mean_price(food)
    # print("HAHA7")
    hotel_mean = mean_price(hotel)
    print("HAHA8")
    return P, food, hotel, total_destination_cost, food_mean, hotel_mean
