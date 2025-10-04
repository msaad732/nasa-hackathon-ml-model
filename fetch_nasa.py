# fetch_nasa.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # optional: loads NASA_API_KEY from .env
API_KEY = os.getenv("NASA_API_KEY")
if not API_KEY:
    raise SystemExit("Set NASA_API_KEY in environment or .env")

OUT_CSV = "data/neo_feed.csv"
os.makedirs("data", exist_ok=True)

def fetch_feed(start_date, end_date):
    url = "https://api.nasa.gov/neo/rest/v1/feed"
    params = {"start_date": start_date, "end_date": end_date, "api_key": API_KEY}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def parse_and_append(json_data, rows):
    neo_dict = json_data.get("near_earth_objects", {})
    for date_str, objs in neo_dict.items():
        for obj in objs:
            try:
                name = obj.get("name")
                diameter_max = obj["estimated_diameter"]["meters"]["estimated_diameter_max"]
                # close approach data may be empty; skip if missing
                cad = obj.get("close_approach_data")
                if not cad:
                    continue
                velocity = float(cad[0]["relative_velocity"]["kilometers_per_second"])
                miss_distance = float(cad[0]["miss_distance"]["kilometers"])
                eccentricity = float(obj.get("orbital_data", {}).get("eccentricity") or 0.0)
                hazardous = bool(obj.get("is_potentially_hazardous_asteroid", False))

                rows.append({
                    "name": name,
                    "date": date_str,
                    "diameter_m": diameter_max,
                    "velocity_km_s": velocity,
                    "miss_distance_km": miss_distance,
                    "eccentricity": eccentricity,
                    "hazardous": int(hazardous)
                })
            except Exception:
                # skip problematic records
                continue

def daterange_chunks(start_date, end_date, chunk_days=7):
    cur = start_date
    while cur <= end_date:
        chunk_end = min(end_date, cur + timedelta(days=chunk_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)

def main():
    # pick your date-range; adjust to collect as many days as you want
    start_date = datetime(2024, 1, 1)   # example start - change as needed
    end_date = datetime(2025, 7, 1)     # example end
    rows = []
    for s, e in tqdm(list(daterange_chunks(start_date, end_date))):
        s_str = s.strftime("%Y-%m-%d")
        e_str = e.strftime("%Y-%m-%d")
        data = fetch_feed(s_str, e_str)
        parse_and_append(data, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
