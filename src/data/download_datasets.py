"""
Download public retail datasets for anomaly detection experiments.

Datasets (all free, no authentication required):
1. M5 Competition (Zenodo) — Walmart daily sales, 3,049 products × 1,969 days
2. Indonesian Food Prices (PIHPS/Bank Indonesia) — 17 commodities × daily
3. Online Retail II (UCI) — e-commerce transactions
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
import zipfile
import requests
from io import BytesIO

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


# ── 1. M5 Competition (from Zenodo) ─────────────────────────────────

def download_m5(force: bool = False) -> pd.DataFrame:
    """
    Download M5 Forecasting Accuracy dataset from Zenodo (free, no auth).

    ~3,049 Walmart products × 1,969 days of daily unit sales.
    """
    dest_dir = RAW_DIR / "m5"
    sales_file = dest_dir / "sales_train_evaluation.csv"

    if sales_file.exists() and not force:
        log.info("M5 data already cached at %s", sales_file)
        return _sample_m5(sales_file)

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip"

    log.info("Downloading M5 from Zenodo (48 MB)...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    log.info("Extracting M5 zip...")
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)

    log.info("M5 extracted to %s", dest_dir)
    return _sample_m5(sales_file)


def _sample_m5(sales_file: Path, n_products: int = 50) -> pd.DataFrame:
    """Read M5 and return long-format daily sales for a product sample."""
    # Load calendar for date mapping
    cal = pd.read_csv(
        sales_file.parent / "calendar.csv",
        usecols=["d", "date"],
        parse_dates=["date"],
    )
    cal["d"] = "d_" + (cal.index + 1).astype(str)

    # Load sales — read first n_products rows
    sales = pd.read_csv(
        sales_file,
        nrows=n_products,
        dtype={
            "item_id": "str",
            "dept_id": "category",
            "cat_id": "category",
            "store_id": "category",
            "state_id": "category",
        },
        low_memory=False,
    )
    sales["id"] = (
        sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str)
    )

    # Melt to long format
    date_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=date_cols,
        var_name="d",
        value_name="sales",
    )

    # Map dates
    long = long.merge(cal, on="d", how="left")
    long = long.dropna(subset=["date"])
    log.info(
        "M5 sample: %d products, %d rows, date range %s to %s",
        long["id"].nunique(), len(long),
        long["date"].min().date(), long["date"].max().date(),
    )
    return long[["date", "id", "sales"]]


# ── 2. Indonesian Food Prices (PIHPS) ──────────────────────────────

def download_indonesian_food_prices(force: bool = False) -> pd.DataFrame:
    """
    Download Indonesian food prices dataset from GitHub.

    Daily prices of essential commodities from PIHPS/Bank Indonesia.
    Contains: rice, chicken, beef, eggs, shallots, garlic, chili, oil, sugar.
    """
    dest = RAW_DIR / "indonesian_food_prices.csv"
    if dest.exists() and not force:
        log.info("Indonesian food prices already cached at %s", dest)
        return _clean_food_prices(dest)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/azzandwi1/"
        "indonesian-food-prices-dataset/main/dataset/daily.csv"
    )

    log.info("Downloading Indonesian food prices from GitHub...")
    df = pd.read_csv(url, dtype=str)  # read all as strings due to commas in numbers
    df["tanggal"] = pd.to_datetime(
        df["tanggal"].str.strip(),
        format="%d/ %m/ %Y",
        errors="coerce",
    )
    # Convert price columns from Indonesian format ("11,050" -> 11050)
    price_cols = [c for c in df.columns if c != "tanggal"]
    for c in price_cols:
        df[c] = pd.to_numeric(
            df[c]
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("-", np.nan),
            errors="coerce",
        )
    df = df.dropna(subset=["tanggal"])
    df.to_csv(dest, index=False)
    log.info("Saved: %s (%d rows, %d columns)", dest, len(df), len(df.columns))
    return _clean_food_prices(dest)


def _clean_food_prices(path: Path) -> pd.DataFrame:
    """Reshape PIHPS data from wide (commodities as columns) to long format."""
    df = pd.read_csv(path, parse_dates=["tanggal"])
    # Melt commodity columns into long format
    id_cols = ["tanggal"]
    value_cols = [c for c in df.columns if c != "tanggal"]
    long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="id",
        value_name="price",
    )
    long.columns = ["date", "id", "sales"]  # rename for unified interface
    long = long.dropna(subset=["sales"])
    long["sales"] = pd.to_numeric(long["sales"], errors="coerce").dropna()
    long = long.sort_values(["id", "date"])
    log.info(
        "Food prices: %d commodities, %d rows, date range %s to %s",
        long["id"].nunique(), len(long),
        long["date"].min().date(), long["date"].max().date(),
    )
    return long


# ── 3. Online Retail II (UCI) ───────────────────────────────────────

def download_online_retail(force: bool = False) -> pd.DataFrame:
    """Download Online Retail II from UCI ML Repository."""
    dest = RAW_DIR / "online_retail_ii.csv"
    if dest.exists() and not force:
        log.info("Online Retail II already cached at %s", dest)
        return _clean_online_retail(dest)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00502/online_retail_II.xlsx"
    )
    log.info("Downloading Online Retail II from UCI...")
    df = pd.read_excel(url)
    df.to_csv(dest, index=False)
    log.info("Saved: %s (%d rows)", dest, len(df))
    return _clean_online_retail(dest)


def _clean_online_retail(path: Path) -> pd.DataFrame:
    """Aggregate to daily sales per stock code."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["date"] = df["InvoiceDate"].dt.date
    daily = (
        df.groupby(["StockCode", "date"])["Quantity"]
        .sum()
        .reset_index()
    )
    daily.columns = ["id", "date", "sales"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["id"] = daily["id"].astype(str)
    daily["sales"] = daily["sales"].clip(lower=0)
    # Top 50 products
    top = daily.groupby("id")["sales"].sum().nlargest(50).index
    daily = daily[daily["id"].isin(top)].copy()
    log.info(
        "Online Retail: %d products, %d rows, date range %s to %s",
        daily["id"].nunique(), len(daily),
        daily["date"].min().date(), daily["date"].max().date(),
    )
    return daily


# ── Unified loader ──────────────────────────────────────────────────

def load_processed_dataset(name: str, n_series: int = 30) -> dict[str, pd.Series]:
    """Load a preprocessed dataset. Returns {id: daily_sales_series}."""
    path = PROCESSED_DIR / f"{name}_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run preprocessing first."
        )
    df = pd.read_parquet(path)
    series_dict = {}
    for sid in df["id"].unique():
        if len(series_dict) >= n_series:
            break
        s = df[df["id"] == sid].set_index("date")["sales"].sort_index()
        if len(s.dropna()) >= 200:  # require at least 200 days
            series_dict[str(sid)] = s
    log.info("Loaded %d/%d series from %s (min 200 days)", len(series_dict), n_series, name)
    return series_dict


if __name__ == "__main__":
    for name, fn in [
        ("M5", lambda: download_m5()),
        ("Food Prices", download_indonesian_food_prices),
        ("Online Retail", lambda: download_online_retail()),
    ]:
        try:
            df = fn()
            log.info("%s: %d rows", name, len(df))
        except Exception as e:
            log.warning("Could not load %s: %s", name, e)
