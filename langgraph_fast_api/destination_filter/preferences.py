import pandas as pd
import numpy as np
import asyncio

# Preferences
def _sync_get_preferences(
        data_terfilter: pd.DataFrame,
        jumlah_hari: int,
        whom: str = "default",
        styles: str | list[str] = ["default"],
) -> tuple[list[float], pd.DataFrame, float]:

    ## With Whom ##
    ### Solo
    if whom == "solo":
        time_limit = [np.random.randint(low=8, high=9) for _ in range(jumlah_hari)]

    ### Family
    elif whom == "family":
        time_limit = [np.random.randint(low=6, high=8) for _ in range(jumlah_hari)]

    ### Couple
    elif whom == "couple":
        time_limit = [np.random.randint(low=9, high=11) for _ in range(jumlah_hari)]

    ### Friends
    elif whom == "friends":
        time_limit = [np.random.randint(low=7, high=10) for _ in range(jumlah_hari)]

    ### Elderly
    elif whom == "elderly":
        time_limit = [np.random.randint(low=5, high=7) for _ in range(jumlah_hari)]

    ### Default (no preferences/flexible)
    else:
        time_limit = [np.random.randint(low=7, high=11) for _ in range(jumlah_hari)]

    ## Travel Style ##
    def choose_style(data_terfilter: pd.DataFrame, styles: str | list[str] = ["default"]):
        if styles is None or "default" in styles:
            return data_terfilter

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

        return data_terfilter[data_terfilter["category"].isin(categories)]

    data_terfilter = choose_style(data_terfilter, styles)

    return time_limit, data_terfilter

async def get_preferences(
        data_terfilter: pd.DataFrame,
        jumlah_hari: int,
        whom: str = "default",
        styles: str | list[str] = ["default"],
) -> tuple[list[float], pd.DataFrame, float]:
    return await asyncio.to_thread(_sync_get_preferences, data_terfilter, jumlah_hari, whom, styles)