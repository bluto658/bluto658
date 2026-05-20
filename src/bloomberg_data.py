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
        logger.info("Loading universe from cache")
        return pd.read_parquet(cache_path)["ticker"].tolist()

    blp = _bbg_import()

    logger.info(f"Fetching index members: {index}")
    members = blp.bds(index, "INDX_MEMBERS")
    raw_tickers = members["member_ticker_and_exch_code"].str.strip().tolist()
    tickers = [f"{t} Equity" for t in raw_tickers]

    logger.info(f"Filtering {len(tickers)} tickers to mkt cap >= ${min_market_cap_mm}M")
    valid: list[str] = []
    for batch in _batch(tickers, 200):
        try:
            caps = blp.bdp(batch, "CUR_MKT_CAP")
            threshold = min_market_cap_mm * 1e6
            above = caps[caps["cur_mkt_cap"].fillna(0) >= threshold].index.tolist()
            valid.extend(above)
        except Exception as exc:
            logger.warning(f"Market cap batch failed, including all: {exc}")
            valid.extend(batch)
        time.sleep(0.3)

    pd.DataFrame({"ticker": valid}).to_parquet(cache_path)
    logger.info(f"Universe: {len(valid)} tickers above ${min_market_cap_mm}M market cap")
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
        logger.info("Loading price history from cache")
        return pd.read_parquet(cache_path)

    blp = _bbg_import()
    per_code = periodicity[0].upper()  # W or M
    all_dfs: list[pd.DataFrame] = []

    for i, batch in enumerate(_batch(tickers, 200)):
        logger.info(f"Price download batch {i + 1} / {len(tickers) // 200 + 1}")
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
            # xbbg returns MultiIndex columns (ticker, field); drop the field level
            raw.columns = raw.columns.droplevel(1)
            all_dfs.append(raw)
        except Exception as exc:
            logger.warning(f"Price batch {i} failed: {exc}")
        time.sleep(0.4)

    if not all_dfs:
        raise RuntimeError("No price data was returned from Bloomberg.")

    prices = pd.concat(all_dfs, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Remove tickers that are mostly empty
    missing = prices.isnull().mean()
    prices = prices.loc[:, missing <= max_missing_pct]
    logger.info(f"Retained {prices.shape[1]} tickers after missing-data filter")

    prices.to_parquet(cache_path)
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
        logger.info("Loading fundamentals from cache")
        return pd.read_parquet(cache_path)

    blp = _bbg_import()
    all_dfs: list[pd.DataFrame] = []

    for i, batch in enumerate(_batch(tickers, 200)):
        logger.info(f"Fundamentals batch {i + 1} / {len(tickers) // 200 + 1}")
        try:
            data = blp.bdp(batch, fields)
            all_dfs.append(data)
        except Exception as exc:
            logger.warning(f"Fundamentals batch {i} failed: {exc}")
        time.sleep(0.3)

    if not all_dfs:
        raise RuntimeError("No fundamental data returned from Bloomberg.")

    fundamentals = pd.concat(all_dfs)
    fundamentals.to_parquet(cache_path)
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

    prices = get_price_history(
        [proxy_ticker], start_date, end_date, periodicity, use_cache
    )
    returns = prices.squeeze().pct_change().dropna()
    returns.name = "market"
    return returns
