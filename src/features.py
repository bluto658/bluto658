"""
Feature engineering for stock cluster analysis.

Four feature families:
  1. Return behavior  – vol, Sharpe, drawdown, skew/kurt
  2. Momentum        – 4w / 13w / 26w / 52w / Jegadeesh-Titman
  3. Factor loadings – market beta + PCA latent factors
  4. Fundamentals    – size, valuation, growth (point-in-time)
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ANNUAL_PERIODS = 52  # weekly data


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Period-over-period returns from a price matrix."""
    return prices.pct_change().dropna(how="all")


def _max_drawdown(s: pd.Series) -> float:
    cum = (1 + s).cumprod()
    peak = cum.expanding().max()
    return float((cum / peak - 1).min())


def _safe_sharpe(ann_ret: float, ann_vol: float, rfr: float = 0.05) -> float:
    return (ann_ret - rfr) / ann_vol if ann_vol > 1e-8 else 0.0


# ---------------------------------------------------------------------------
# Feature family 1: Return behavior
# ---------------------------------------------------------------------------

def compute_return_behavior(returns: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    """
    Annualized vol, Sharpe, max drawdown, skewness, kurtosis, total return.
    Uses the most recent `window` observations.
    """
    r = returns.tail(window)
    records: dict[str, dict] = {}

    for col in r.columns:
        s = r[col].dropna()
        if len(s) < window // 3:
            continue

        total_ret = float((1 + s).prod() - 1)
        ann_vol = float(s.std() * np.sqrt(ANNUAL_PERIODS))
        ann_ret = float((1 + total_ret) ** (ANNUAL_PERIODS / len(s)) - 1)

        records[col] = {
            "total_return": total_ret,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": _safe_sharpe(ann_ret, ann_vol),
            "max_drawdown": _max_drawdown(s),
            "return_skew": float(stats.skew(s)),
            "return_kurt": float(stats.kurtosis(s)),
            "pct_positive_weeks": float((s > 0).mean()),
        }

    return pd.DataFrame(records).T


# ---------------------------------------------------------------------------
# Feature family 2: Momentum
# ---------------------------------------------------------------------------

def compute_momentum(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Total return over 4w, 13w, 26w, 52w windows.
    Also computes Jegadeesh-Titman momentum (12m ex last month).
    All computed relative to the last available date.
    """
    lookbacks = {"mom_4w": 4, "mom_13w": 13, "mom_26w": 26, "mom_52w": 52}
    records: dict[str, dict] = {}

    for col in returns.columns:
        s = returns[col].dropna()
        row: dict[str, float] = {}

        for name, w in lookbacks.items():
            if len(s) >= w:
                row[name] = float((1 + s.tail(w)).prod() - 1)
            else:
                row[name] = float("nan")

        # Jegadeesh-Titman: 52w total return excluding most recent 4w
        if len(s) >= 52:
            row["mom_jt"] = float((1 + s.tail(52).head(48)).prod() - 1)
        else:
            row["mom_jt"] = float("nan")

        records[col] = row

    return pd.DataFrame(records).T


# ---------------------------------------------------------------------------
# Feature family 3: Factor exposures
# ---------------------------------------------------------------------------

def compute_factor_exposures(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 52,
    n_pca_factors: int = 3,
) -> pd.DataFrame:
    """
    Estimate each stock's loading on:
      - The market factor (SPY excess return)
      - Top `n_pca_factors` PCA factors extracted from the full return matrix

    Uses OLS over the rolling window.
    """
    r = returns.tail(window)
    mkt = market_returns.reindex(r.index).fillna(0)

    # Extract PCA latent factors from the cross-sectional returns
    r_filled = r.fillna(r.median())
    n_components = min(n_pca_factors, r_filled.shape[1] - 1, r_filled.shape[0] - 1)
    pca_factors: Optional[pd.DataFrame] = None

    if n_components > 0 and r_filled.shape[1] > n_pca_factors:
        pca = PCA(n_components=n_components, random_state=42)
        # Fit on stocks (columns); transform gives factors as time series
        factor_matrix = pca.fit_transform(r_filled.T).T  # (n_factors, n_dates)
        pca_factors = pd.DataFrame(
            factor_matrix.T,
            index=r.index,
            columns=[f"pca_f{i + 1}" for i in range(n_components)],
        )
        logger.debug(f"PCA explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.1%}")

    # Build the regressor matrix
    X_parts = [mkt.rename("market")]
    if pca_factors is not None:
        X_parts.append(pca_factors)
    X = pd.concat(X_parts, axis=1).dropna()
    X_arr = X.values
    factor_cols = X.columns.tolist()

    records: dict[str, dict] = {}
    for col in r.columns:
        y = r[col].reindex(X.index).fillna(0).values
        if y.std() < 1e-8:
            continue
        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_arr, y)
            row: dict[str, float] = {
                "alpha_ann": float(reg.intercept_) * ANNUAL_PERIODS,
                "r_squared": float(reg.score(X_arr, y)),
            }
            for i, fname in enumerate(factor_cols):
                row[f"beta_{fname}"] = float(reg.coef_[i])
        except Exception:
            row = {"alpha_ann": 0.0, "r_squared": 0.0, "beta_market": 1.0}

        records[col] = row

    return pd.DataFrame(records).T


# ---------------------------------------------------------------------------
# Feature family 4: Fundamentals
# ---------------------------------------------------------------------------

def prepare_fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean fundamental columns for clustering.
    Applies log transform to market cap; winsorizes valuation ratios.
    """
    col_map = {
        "cur_mkt_cap": "log_mkt_cap",
        "pe_ratio": "pe_ratio",
        "sales_5yr_growth_rate": "sales_growth",
        "dvd_yild": "div_yield",
        "px_to_book_ratio": "price_to_book",
        "tot_debt_to_tot_eqy": "debt_to_equity",
    }

    parts = {}
    for src, dst in col_map.items():
        if src not in fundamentals.columns:
            continue
        s = fundamentals[src].copy()

        if dst == "log_mkt_cap":
            s = np.log1p(s.clip(lower=0))
        else:
            # Winsorize at 2nd/98th percentile
            lo, hi = s.quantile(0.02), s.quantile(0.98)
            s = s.clip(lo, hi)

        parts[dst] = s

    return pd.DataFrame(parts)


# ---------------------------------------------------------------------------
# Combined feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    fundamentals: Optional[pd.DataFrame] = None,
    window: int = 52,
    n_pca_factors: int = 3,
) -> pd.DataFrame:
    """
    Combine all four feature families into a single, aligned DataFrame.
    Rows are tickers; columns are features.
    """
    rb = compute_return_behavior(returns, window)
    mom = compute_momentum(returns.tail(window))
    fexp = compute_factor_exposures(returns, market_returns, window, n_pca_factors)

    common = rb.index.intersection(mom.index).intersection(fexp.index)
    feat = pd.concat([rb.loc[common], mom.loc[common], fexp.loc[common]], axis=1)

    if fundamentals is not None:
        fund = prepare_fundamental_features(fundamentals)
        feat = feat.join(fund.reindex(common), how="left")

    # Drop columns that are all NaN, then fill residual NaN with column median
    feat = feat.dropna(axis=1, how="all")
    feat = feat.fillna(feat.median())

    # Final winsorize at 1st/99th percentile across the merged matrix
    num_cols = feat.select_dtypes(include=[np.number]).columns
    lo = feat[num_cols].quantile(0.01)
    hi = feat[num_cols].quantile(0.99)
    feat[num_cols] = feat[num_cols].clip(lo, hi, axis=1)

    return feat


def normalize_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Robust-scale numeric features (median=0, IQR=1)."""
    num = feat.select_dtypes(include=[np.number])
    scaled = RobustScaler().fit_transform(num)
    return pd.DataFrame(scaled, index=feat.index, columns=num.columns)
