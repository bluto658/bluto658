"""
Bloomberg data fetching with caching.
Requires a live Bloomberg Terminal connection and xbbg installed:
    pip install xbbg blpapi
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FUNDAMENTAL_FIELDS = [
    "CUR_MKT_CAP",            # market cap in USD
    "PE_RATIO",               # trailing P/E
    "SALES_5YR_GROWTH_RATE",  # 5-year revenue CAGR
    "DVD_YILD",               # dividend yield
    "PX_TO_BOOK_RATIO",       # price/book (value proxy)
    "TOT_DEBT_TO_TOT_EQY",    # leverage
    "GICS_SECTOR_NAME",       # GICS sector
    "GICS_INDUSTRY_NAME",     # GICS industry
    "SHORT_NAME",             # company short name
    "CNTRY_OF_INCORPORATION", # domicile
]


def _cache(name: str) -> Path:
    return RAW_DIR / f"{name}.parquet"


def _is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    age_secs = datetime.now().timestamp() - path.stat().st_mtime
    return age_secs < max_age_days * 86400


def _bbg_import():
    try:
        from xbbg import blp
        return blp
    except ImportError as e:
        raise ImportError(
            "xbbg not installed. Run: pip install xbbg\n"
            "Also requires blpapi from the Bloomberg SDK."
        ) from e


def _batch(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def test_connection() -> bool:
    """
    Quick Bloomberg connectivity check.
    Fetches one field for SPY and prints a clear pass/fail message.
    Call this before any data download to confirm Terminal is connected.
    """
    print("Testing Bloomberg connection...", flush=True)
    try:
        blp = _bbg_import()
        result = blp.bdp(["SPY US Equity"], "SHORT_NAME")
        name = result["short_name"].iloc[0]
        print(f"  Bloomberg connected — SPY returned: '{name}'")
        return True
    except ImportError as e:
        print(f"  FAILED — xbbg not installed: {e}")
        return False
    except Exception as e:
        print(f"  FAILED — could not reach Bloomberg Terminal: {e}")
        print("  Make sure Bloomberg Terminal is open and you are logged in.")
        return False


def get_universe(
    index: str = "RAY Index",
    min_market_cap_mm: float = 100,
    use_cache: bool = True,
    cache_days: int = 7,
) -> list[str]:
    """
    Pull all members of a Bloomberg equity index and filter by market cap.
    Default index is Russell 3000 (RAY Index).

    Returns a list of Bloomberg equity tickers, e.g. ['AAPL US Equity', ...].
    """
    cache_path = _cache(f"universe_{index.replace(' ', '_')}_mc{int(min_market_cap_mm)}")
    if use_cache and _is_fresh(cache_path, cache_days):
        tickers = pd.read_parquet(cache_path)["ticker"].tolist()
        print(f"  Universe loaded from cache: {len(tickers):,} tickers")
        return tickers

    blp = _bbg_import()

    print(f"Fetching index members: {index} ...", flush=True)
    members = blp.bds(index, "INDX_MEMBERS")
    raw_tickers = members["member_ticker_and_exch_code"].str.strip().tolist()
    tickers = [f"{t} Equity" for t in raw_tickers]
    print(f"  {len(tickers):,} index members found")

    print(f"Filtering to market cap >= ${min_market_cap_mm}M ...", flush=True)
    batches = list(_batch(tickers, 200))
    valid: list[str] = []

    with tqdm(batches, desc="  Market cap filter", unit="batch") as pbar:
        for batch in pbar:
            try:
                caps = blp.bdp(batch, "CUR_MKT_CAP")
                threshold = min_market_cap_mm * 1e6
                above = caps[caps["cur_mkt_cap"].fillna(0) >= threshold].index.tolist()
                valid.extend(above)
                pbar.set_postfix({"kept": len(valid)})
            except Exception as exc:
                logger.warning(f"Market cap batch failed, including all: {exc}")
                valid.extend(batch)
            time.sleep(0.3)

    pd.DataFrame({"ticker": valid}).to_parquet(cache_path)
    print(f"  Universe: {len(valid):,} tickers above ${min_market_cap_mm}M  (saved to cache)")
    return valid


def get_price_history(
    tickers: list[str],
    start_date: str,
    end_date: str,
    periodicity: str = "WEEKLY",
    use_cache: bool = True,
    max_missing_pct: float = 0.30,
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Bloomberg.
    Batches requests to avoid overloading the API.

    Returns DataFrame: DatetimeIndex (dates) x tickers.
    Drops tickers with > max_missing_pct missing observations.
    """
    safe_key = f"prices_{periodicity}_{start_date}_{end_date}_{len(tickers)}t"
    cache_path = _cache(safe_key)
    if use_cache and cache_path.exists():
        prices = pd.read_parquet(cache_path)
        print(f"  Prices loaded from cache: {prices.shape[0]} weeks x {prices.shape[1]:,} tickers")
        return prices

    blp = _bbg_import()
    per_code = periodicity[0].upper()  # W or M
    all_dfs: list[pd.DataFrame] = []
    batches = list(_batch(tickers, 200))

    print(f"Downloading {periodicity.lower()} prices: {start_date} → {end_date}", flush=True)
    print(f"  {len(tickers):,} tickers  |  {len(batches)} batches of 200")

    with tqdm(batches, desc="  Price download", unit="batch") as pbar:
        for batch in pbar:
            try:
                raw = blp.bdh(
                    batch,
                    "PX_LAST",
                    start_date,
                    end_date,
                    Per=per_code,
                    Fill="P",           # carry forward on non-trading days
                    CshAdjNormal=True,  # adjust for regular dividends/splits
                    CshAdjAbnormal=True,
                )
                raw.columns = raw.columns.droplevel(1)
                all_dfs.append(raw)
                pbar.set_postfix({"tickers_so_far": sum(d.shape[1] for d in all_dfs)})
            except Exception as exc:
                logger.warning(f"Price batch failed: {exc}")
            time.sleep(0.4)

    if not all_dfs:
        raise RuntimeError("No price data was returned from Bloomberg.")

    prices = pd.concat(all_dfs, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    missing = prices.isnull().mean()
    prices = prices.loc[:, missing <= max_missing_pct]

    prices.to_parquet(cache_path)
    print(f"  Done: {prices.shape[0]} weeks x {prices.shape[1]:,} tickers  (saved to cache)")
    return prices


def get_fundamentals(
    tickers: list[str],
    fields: list[str] | None = None,
    use_cache: bool = True,
    cache_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch current fundamental snapshot for all tickers.
    Returns DataFrame: tickers x fields.
    """
    if fields is None:
        fields = FUNDAMENTAL_FIELDS

    cache_path = _cache(f"fundamentals_{len(tickers)}t")
    if use_cache and _is_fresh(cache_path, cache_days):
        df = pd.read_parquet(cache_path)
        print(f"  Fundamentals loaded from cache: {df.shape[0]:,} tickers x {df.shape[1]} fields")
        return df

    blp = _bbg_import()
    all_dfs: list[pd.DataFrame] = []
    batches = list(_batch(tickers, 200))

    print(f"Downloading fundamentals for {len(tickers):,} tickers ...", flush=True)

    with tqdm(batches, desc="  Fundamentals", unit="batch") as pbar:
        for batch in pbar:
            try:
                data = blp.bdp(batch, fields)
                all_dfs.append(data)
            except Exception as exc:
                logger.warning(f"Fundamentals batch failed: {exc}")
            time.sleep(0.3)

    if not all_dfs:
        raise RuntimeError("No fundamental data returned from Bloomberg.")

    fundamentals = pd.concat(all_dfs)
    fundamentals.to_parquet(cache_path)
    print(f"  Done: {fundamentals.shape[0]:,} tickers  (saved to cache)")
    return fundamentals


def get_market_factor(
    proxy_ticker: str = "SPY US Equity",
    start_date: str = "20220101",
    end_date: str | None = None,
    periodicity: str = "WEEKLY",
    use_cache: bool = True,
) -> pd.Series:
    """Download returns for a market proxy (default: SPY)."""
    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    print(f"Fetching market factor ({proxy_ticker}) ...", flush=True)
    prices = get_price_history(
        [proxy_ticker], start_date, end_date, periodicity, use_cache
    )
    returns = prices.squeeze().pct_change().dropna()
    returns.name = "market"
    print(f"  Market factor: {len(returns)} weekly observations")
    return returns
