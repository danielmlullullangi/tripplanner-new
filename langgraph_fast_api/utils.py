from typing_extensions import Any
import numpy as np
import pandas as pd

def scaler(
        array: np.array,
        scale: int
):
    return np.round(array * scale).astype(int)

def generate_metadata(
        res: dict[int, list[str]],
        query: pd.DataFrame
) -> dict[int, dict[str, dict[str, Any]]]:

    cols = [
        "poi", "body", "url_image", "link_drive", "price_min", "price_mean", "price_max", "howtoget_there",
        "tips", "phone", "address", "website_sosmed",
        "latitude", "longitude", "rating", "rating_count"
    ]

    query_indexed = query.set_index("title")

    result = {}

    for key, values in res.items():
        names = {}

        for value in values:
            if value not in query_indexed.index:
                continue

            row = query_indexed.loc[value, cols]

            # Jika ada duplicate title, ambil baris pertama
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            metadata = row.to_dict()
            names[value] = metadata

        result[key] = names

    return result