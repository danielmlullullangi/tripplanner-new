import pandas as pd
import numpy as np
import asyncio

# Preferences
def _sync_get_preferences(
        query: pd.DataFrame,
        D: int,
        whom: str = "default",
        styles: str | list[str] = ["default"],
) -> tuple[list[float], pd.DataFrame, float]:

    ## With Whom ##
    ### Solo
    if whom == "solo":
        time_limit = [np.random.randint(low=8, high=9) for _ in range(D)]

    ### Family
    elif whom == "family":
        time_limit = [np.random.randint(low=6, high=8) for _ in range(D)]

    ### Couple
    elif whom == "couple":
        time_limit = [np.random.randint(low=9, high=11) for _ in range(D)]

    ### Friends
    elif whom == "friends":
        time_limit = [np.random.randint(low=7, high=10) for _ in range(D)]

    ### Elderly
    elif whom == "elderly":
        time_limit = [np.random.randint(low=5, high=7) for _ in range(D)]

    ### Default (no preferences/flexible)
    else:
        time_limit = [np.random.randint(low=7, high=11) for _ in range(D)]

    ## Travel Style ##
    def choose_style(query: pd.DataFrame, styles: str | list[str] = ["default"]):
        if styles is None or "default" in styles:
            return query

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

        categories = set()
        for style in styles:
            categories.update(style_map.get(style, []))

        return query[query["category"].isin(categories)]

    query = choose_style(query, styles)

    return time_limit, query

async def get_preferences(
        query: pd.DataFrame,
        D: int,
        whom: str = "default",
        styles: str | list[str] = ["default"],
) -> tuple[list[float], pd.DataFrame, float]:
    return await asyncio.to_thread(_sync_get_preferences, query, D, whom, styles)