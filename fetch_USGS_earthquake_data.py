import requests
import pandas as pd

def fetch_usgs_earthquakes(start="2025-01-01", end="2025-10-01", min_magnitude=5):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_magnitude,
        "limit": 20000  # max allowed
    }
    resp = requests.get(url, params=params).json()

    records = []
    for feature in resp["features"]:
        props = feature["properties"]
        records.append({
            "time": props["time"],
            "place": props["place"],
            "magnitude": props["mag"],
            "title": props["title"]
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = fetch_usgs_earthquakes()
    print(df.head())
    df.to_csv("data/usgs_earthquakes.csv", index=False)
