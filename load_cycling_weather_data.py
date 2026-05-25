"""
load_cycling_weather.py
=======================
Helpers for loading the joined AWV cycling × RMI weather dataset.

Quick start
-----------
1. Save `cycling_weather_full.parquet` somewhere on your computer.
2. Update DATA_PATH below to that location.
3. Use one of the loaders:

    from load_cycling_weather import (
        load_full,             # everything (~40M rows, big!)
        load_columns,          # only the columns you need
        load_date_range,       # only certain months
        load_one_site,         # only one cycling site
        load_sample,           # quick random sample for prototyping
        peek,                  # metadata only, no data loaded
    )

    # Example — quick look at the data without loading anything heavy
    peek()

    # Example — load only the rows for a typical regression
    df = load_columns(["ts", "site_id", "count",
                       "precip_quantity", "temp_dry_shelter_avg"])

    # Example — work with a 1% sample while iterating on code
    df = load_sample(frac=0.01)

    # Example — focus on summer 2024
    df = load_date_range("2024-06-01", "2024-09-01")

Requirements
------------
    pip install pandas pyarrow
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================
#                ✦  EDIT THIS PATH  ✦
# =============================================================
# Point this to wherever you saved the parquet file.
# Examples:
#   Windows:  Path(r"C:\Users\yourname\Downloads\cycling_weather_full.parquet")
#   Mac:      Path("/Users/yourname/data/cycling_weather_full.parquet")
#   Linux:    Path("/home/yourname/data/cycling_weather_full.parquet")
DATA_PATH = Path(r"C:\Users\User\Desktop\KU Leuven. Msc Statistics. Year 1\Modern Data Analytics\cycling_weather_full.parquet")


# =============================================================
#                       LOADERS
# =============================================================

def peek():
    """Print metadata about the file without loading any rows."""
    _check_path()
    pf = pq.ParquetFile(DATA_PATH)
    print(f"File:    {DATA_PATH}")
    print(f"Size:    {DATA_PATH.stat().st_size / 1e6:.1f} MB on disk")
    print(f"Rows:    {pf.metadata.num_rows:,}")
    print(f"Columns: {pf.schema_arrow.names}")


def load_full() -> pd.DataFrame:
    """
    Load the entire dataset (≈40M rows, ~6 GB in RAM).
    Only use this if you have plenty of memory — see load_columns or
    load_sample for lighter alternatives.
    """
    _check_path()
    return pd.read_parquet(DATA_PATH)


def load_columns(columns: list[str]) -> pd.DataFrame:
    """
    Load all rows but only the columns you ask for.
    Much lighter than load_full(): if you only need 5 of the 19 columns,
    memory drops by roughly 5/19.
    """
    _check_path()
    return pd.read_parquet(DATA_PATH, columns=columns)


def load_date_range(start: str, end: str,
                    columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load only rows whose timestamp falls in [start, end).

    Dates can be any string pandas understands, e.g. "2024-06-01"
    or "2024-06-01 12:00".

    The function reads the file in chunks and keeps only matching rows,
    so memory stays low even for big ranges.
    """
    _check_path()
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")

    pf = pq.ParquetFile(DATA_PATH)
    cols = columns if columns else None
    if cols and "ts" not in cols:
        cols = list(cols) + ["ts"]   # need ts to filter

    pieces = []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        b = batch.to_pandas()
        mask = (b["ts"] >= start_ts) & (b["ts"] < end_ts)
        if mask.any():
            pieces.append(b.loc[mask])
    if not pieces:
        return pd.DataFrame(columns=cols or pf.schema_arrow.names)
    out = pd.concat(pieces, ignore_index=True)
    if columns and "ts" not in columns:
        out = out.drop(columns=["ts"])
    return out


def load_one_site(site_id: int,
                  columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load all rows for a single cycling site. Fast and small —
    one site has ≈250k rows max, fits comfortably in memory.
    """
    _check_path()
    pf = pq.ParquetFile(DATA_PATH)
    cols = columns if columns else None
    if cols and "site_id" not in cols:
        cols = list(cols) + ["site_id"]

    pieces = []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        b = batch.to_pandas()
        mask = b["site_id"] == site_id
        if mask.any():
            pieces.append(b.loc[mask])
    if not pieces:
        return pd.DataFrame(columns=cols or pf.schema_arrow.names)
    out = pd.concat(pieces, ignore_index=True)
    if columns and "site_id" not in columns:
        out = out.drop(columns=["site_id"])
    return out


def load_sample(frac: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """
    Load a random fraction of rows. Useful while writing/debugging
    code so iteration is fast — switch to load_full() or
    load_columns() once your code works.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    _check_path()
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(DATA_PATH)
    pieces = []
    for batch in pf.iter_batches(batch_size=200_000):
        b = batch.to_pandas()
        keep = rng.random(len(b)) < frac
        if keep.any():
            pieces.append(b.loc[keep])
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# =============================================================
#                  COLUMN REFERENCE
# =============================================================
"""
Columns in the parquet file
---------------------------
site_id                  int     AWV cycling site identifier
direction                str     'IN' or 'OUT' relative to the counter
mode                     str     'FIETSERS' (cyclists) — already filtered
interval_start           str     start of the 15-min cycling interval
interval_end             str     end of the 15-min cycling interval
count                    float   cyclists counted in the 15-min interval
ts                       ts UTC  same as interval_end, parsed to datetime
nearest_station_code     int     RMI weather station matched to this site
nearest_station_km       float   distance site→station in km
site_lat, site_lon       float   site coordinates
precip_quantity          float   mm of rain in the 15-min interval (sum)
sun_duration             float   minutes of sun in the 15-min interval (sum)
temp_dry_shelter_avg     float   air temperature °C (mean of two 10-min readings)
wind_speed_10m           float   m/s at 10 m altitude (mean)
wind_gusts_speed         float   m/s peak gust (max)
humidity_rel_shelter_avg float   relative humidity % (mean)
pressure                 float   hPa at station level (mean)
short_wave_from_sky_avg  float   global radiation W/m² (mean)
"""


# =============================================================
#                       INTERNAL
# =============================================================
def _check_path():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n\nFile not found: {DATA_PATH}\n"
            f"Edit DATA_PATH at the top of load_cycling_weather.py "
            f"to point at your local copy of cycling_weather_full.parquet.\n"
        )


if __name__ == "__main__":
    peek()
