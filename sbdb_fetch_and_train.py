import requests
import pandas as pd

def fetch_sbdb(limit=20):
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    query = {
        "fields": ["object", "pha", "H", "diameter", "albedo"],
        "limit": limit
    }

    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, json=query, headers=headers).json()

    if "data" not in resp:
        raise ValueError(f"Unexpected API response: {resp}")

    df = pd.DataFrame(resp["data"], columns=resp["fields"])
    return df

if __name__ == "__main__":
    df = fetch_sbdb(limit=10)
    print(df.head())
    df.to_csv("data/sbdb_data.csv", index=False)
