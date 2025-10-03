#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latvia Housing Dashboard: Fetch + Clean + Merge
------------------------------------------------
What it does:
- Pulls Eurostat data via the "Statistics API" (JSON-stat 2.0):
  * Quarterly House Price Index (PRC_HPI_Q) for LV
  * Monthly HICP CP041 (Actual rentals for housing) for LV (aggregated to annual)
  * Quarterly Building Permits (STS_COBP_Q) for LV
- Reads CSB regional CSVs that you export manually:
  * Wages by region and year (EUR)  --> avg_monthly_wage
  * Building permits by region/year --> building_permits (optional; preferred over national when present)
  * Households by region/year       --> households
  * Housing stock by region/year    --> housing_stock (Census baseline; can be static 2021)
- Produces ONE unified CSV shaped like your mock file:
  latvia,year,region,price_per_m2,avg_monthly_wage,avg_monthly_rent,building_permits,households,housing_stock

How to run (locally):
    python fetch_and_clean_data.py

Requirements:
    pip install pandas requests pyjstat

References / API format:
- Eurostat "Statistics API" base:
  https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{DATASET}?filters...
  (see: API - Detailed guidelines - API Statistics)
"""
import os
import io
import sys
import json
import math
import time
import zipfile
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from pyjstat import pyjstat

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# ---- CONFIG ----
OUT_DIR = "out"
RAW_DIR = "raw"   # You can drop CSB CSVs here
REGIONS = ["Riga", "Pieriga", "Liepaja"]   # target regions for the unified file
YEARS   = list(range(2021, 2025))          # 2021-2024
COUNTRY = "LV"

# If you have a known €/m2 anchor for a region/year, put it here to scale HPI index to price_per_m2.
# Example: {'2023:Riga': 2100, '2024:Riga': 2270}
PRICE_ANCHORS = {}

# Rent scaling ratios by region (use if only national CP041 exists)
# You can tune these later based on market intel
RENT_SCALE = {
    "Riga": 1.20,
    "Pieriga": 1.10,
    "Liepaja": 0.80
}

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

def eurostat_fetch_json(dataset: str, params: Dict[str, str]) -> dict:
    """Fetch JSON-stat data from Eurostat Statistics API."""
    url = f"{EUROSTAT_BASE}/{dataset}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def eurostat_to_df(json_obj: dict) -> pd.DataFrame:
    """Convert JSON-stat to pandas DataFrame."""
    try:
        dataset = pyjstat.Dataset.read(json_obj)
        df = dataset.write('dataframe')
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON-stat: {e}")

def get_hpi_quarterly_lv(since="2015-Q1") -> pd.DataFrame:
    """
    PRC_HPI_Q - House price index (2015=100), quarterly, Latvia.
    Returns a DataFrame with columns: time, value
    """
    params = {
        "geo": COUNTRY,
        "sinceTimePeriod": since,
        "lang": "EN",
        "format": "JSON"
    }
    js = eurostat_fetch_json("PRC_HPI_Q", params)
    df = eurostat_to_df(js)
    # Expect columns like: time, geo, unit, value
    df = df.rename(columns=str.lower)
    df = df[df.get("geo","") == COUNTRY]
    df = df[["time", "value"]].copy()
    # Extract year and quarter
    df["year"] = df["time"].str.slice(0,4).astype(int)
    # Average quarterly index to annual index
    annual = df.groupby("year", as_index=False)["value"].mean().rename(columns={"value":"hpi_index"})
    return annual

def get_hicp_cp041_lv(since="2015-01") -> pd.DataFrame:
    """
    PRC_HICP_MIDX - HICP monthly index. Filter to CP041 (Actual rentals for housing), Latvia.
    Aggregates to annual average index.
    """
    params = {
        "geo": COUNTRY,
        "coicop": "CP041",
        "sinceTimePeriod": since,
        "lang": "EN",
        "format": "JSON"
    }
    js = eurostat_fetch_json("PRC_HICP_MIDX", params)
    df = eurostat_to_df(js)
    df = df.rename(columns=str.lower)
    df = df[(df.get("geo","") == COUNTRY) & (df.get("coicop","") == "CP041")]
    # time looks like YYYY-MM
    df["year"] = df["time"].str.slice(0,4).astype(int)
    annual = df.groupby("year", as_index=False)["value"].mean().rename(columns={"value":"rent_index"})
    return annual

def get_building_permits_lv(since="2015-Q1") -> pd.DataFrame:
    """
    STS_COBP_Q - Building permits (m2 of useful floor area), quarterly, Latvia.
    Aggregates to annual total index/value.
    """
    params = {
        "geo": COUNTRY,
        "sinceTimePeriod": since,
        "lang": "EN",
        "format": "JSON"
    }
    js = eurostat_fetch_json("STS_COBP_Q", params)
    df = eurostat_to_df(js)
    df = df.rename(columns=str.lower)
    df = df[df.get("geo","") == COUNTRY]
    # sum or average? Permits are typically summed over quarters in a year
    df["year"] = df["time"].str.slice(0,4).astype(int)
    annual = df.groupby("year", as_index=False)["value"].sum().rename(columns={"value":"permits_national"})
    return annual

def read_csb_regional_csv(filename: str, col_map: Dict[str,str], region_col: str, year_col: str, value_col: str) -> pd.DataFrame:
    """
    Generic reader for CSB CSV exports you download manually.
    Provide a column mapper to standardize names.
    """
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        warnings.warn(f"CSB file missing: {path}")
        return pd.DataFrame(columns=["region","year",col_map.get(value_col,"value")])
    df = pd.read_csv(path)
    # Standardize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rcol = region_col.lower()
    ycol = year_col.lower()
    vcol = value_col.lower()
    # Basic clean
    out = df[[rcol, ycol, vcol]].copy()
    out.columns = ["region", "year", col_map.get(value_col, "value")]
    # Normalize region spellings
    out["region"] = out["region"].replace({
        "rīga": "Riga",
        "riga": "Riga",
        "pierīga": "Pieriga",
        "pieriga": "Pieriga",
        "liepāja": "Liepaja",
        "liepaja": "Liepaja"
    })
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    # Keep only our target regions and years
    out = out[out["region"].isin(REGIONS) & out["year"].isin(YEARS)]
    return out

def scale_index_to_price_per_m2(hpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert HPI index to approximate €/m2 using PRICE_ANCHORS if provided.
    If no anchors, we map the national index to a nominal price series by setting
    the first available year to an arbitrary 1000 €/m2 and scaling forward.
    """
    df = hpi_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["year","price_per_m2_base"])
    df = df.sort_values("year")
    # baseline
    base_level = 1000.0
    base_index = df.iloc[0]["hpi_index"]
    df["price_per_m2_base"] = base_level * (df["hpi_index"] / base_index)
    return df[["year","price_per_m2_base"]]

def build_unified():
    ensure_dirs()
    # ---- Fetch Eurostat national series ----
    print("Fetching Eurostat HPI (quarterly -> annual)...")
    hpi = get_hpi_quarterly_lv(since="2015-Q1")
    print("Fetching Eurostat HICP CP041 (monthly -> annual)...")
    rent = get_hicp_cp041_lv(since="2015-01")
    print("Fetching Eurostat Building Permits (quarterly -> annual)...")
    perm_nat = get_building_permits_lv(since="2015-Q1")

    # Derive base €/m2 from HPI index (national), then copy per region (until we have CSB regional prices)
    price_base = scale_index_to_price_per_m2(hpi)  # year, price_per_m2_base

    # ---- Read CSB regional CSVs you export manually ----
    # 1) Wages by region: drop a CSV in raw/ named 'csb_wages_by_region.csv' with columns like: Region, Year, Value
    wages = read_csb_regional_csv(
        filename="csb_wages_by_region.csv",
        col_map={"value":"avg_monthly_wage"},
        region_col="Region",
        year_col="Year",
        value_col="Value"
    )

    # 2) Regional permits (optional, preferred over national if provided)
    permits_reg = read_csb_regional_csv(
        filename="csb_permits_by_region.csv",
        col_map={"value":"building_permits"},
        region_col="Region",
        year_col="Year",
        value_col="Value"
    )

    # 3) Households by region
    households = read_csb_regional_csv(
        filename="csb_households_by_region.csv",
        col_map={"value":"households"},
        region_col="Region",
        year_col="Year",
        value_col="Value"
    )

    # 4) Housing stock by region (Census)
    housing_stock = read_csb_regional_csv(
        filename="csb_housing_stock_by_region.csv",
        col_map={"value":"housing_stock"},
        region_col="Region",
        year_col="Year",
        value_col="Value"
    )

    # ---- Assemble the unified frame ----
    rows = []
    for y in YEARS:
        # get national values for the year
        p = price_base.loc[price_base["year"] == y, "price_per_m2_base"]
        p = float(p.iloc[0]) if len(p) else None

        r_index = rent.loc[rent["year"] == y, "rent_index"]
        rent_national = float(r_index.iloc[0]) if len(r_index) else None

        nat_permits = perm_nat.loc[perm_nat["year"] == y, "permits_national"]
        nat_permits = float(nat_permits.iloc[0]) if len(nat_permits) else None

        for region in REGIONS:
            # Price anchors (optional, if provided)
            key = f"{y}:{region}"
            price_per_m2 = PRICE_ANCHORS.get(key, p)

            # Wages (regional)
            w = wages[(wages["year"] == y) & (wages["region"] == region)]
            wage_val = float(w["avg_monthly_wage"].iloc[0]) if len(w) else None

            # Rents (estimate from national CP041 using scaling ratios)
            rent_est = None
            if rent_national is not None:
                scale = RENT_SCALE.get(region, 1.0)
                # If you later set a real anchor, swap this with real EUR/month
                rent_est = rent_national * scale

            # Permits (regional if present, else fallback to share of national)
            pr = permits_reg[(permits_reg["year"] == y) & (permits_reg["region"] == region)]
            if len(pr):
                permits_val = float(pr["building_permits"].iloc[0])
            else:
                # Fallback: distribute national permits equally (or weighted by households)
                permits_val = None
                if nat_permits is not None:
                    # try to weight by households share
                    hh_year = households[households["year"] == y]
                    if len(hh_year) and hh_year["households"].sum() > 0:
                        hh_region = hh_year[hh_year["region"] == region]["households"]
                        if len(hh_region):
                            share = float(hh_region.iloc[0]) / float(hh_year["households"].sum())
                            permits_val = nat_permits * share
                    if permits_val is None:
                        permits_val = nat_permits / len(REGIONS)

            # Households & stock
            hh = households[(households["year"] == y) & (households["region"] == region)]
            hh_val = float(hh["households"].iloc[0]) if len(hh) else None

            hs = housing_stock[(housing_stock["year"] == y) & (housing_stock["region"] == region)]
            hs_val = float(hs["housing_stock"].iloc[0]) if len(hs) else None

            rows.append({
                "latvia": COUNTRY,
                "year": y,
                "region": region,
                "price_per_m2": None if price_per_m2 is None else round(price_per_m2, 2),
                "avg_monthly_wage": wage_val,
                "avg_monthly_rent": None if rent_est is None else round(rent_est, 2),
                "building_permits": None if permits_val is None else round(permits_val, 2),
                "households": hh_val,
                "housing_stock": hs_val
            })

    unified = pd.DataFrame(rows)

    out_csv = os.path.join(OUT_DIR, "latvia_housing_unified.csv")
    unified.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    ensure_dirs()
    build_unified()
