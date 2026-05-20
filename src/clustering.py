"""
Clustering algorithms and temporal peer analysis.

Key capabilities:
  - K-Means and hierarchical clustering
  - Silhouette-based k selection
  - Rolling window clustering with stable label alignment (Hungarian matching)
  - Peer finding with return-correlation ranking
  - Peer history / cluster drift tracking
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# k selection
# ---------------------------------------------------------------------------

def score_k_range(
    X: np.ndarray,
    k_range: range = range(5, 51, 5),
    sample_size: int = 3000,
    random_state: int = 42,
) -> dict[int, float]:
    """
    Evaluate K-Means silhouette score for each k.
    Returns {k: silhouette_score}.
    """
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X_s = X[idx]
    else:
        X_s = X

    scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_s)
        s = silhouette_score(X_s, labels, sample_size=min(2000, len(X_s)))
        scores[k] = round(float(s), 5)
        logger.info(f"  k={k:3d}  silhouette={s:.4f}")
    return scores


def best_k(scores: dict[int, float]) -> int:
    """Return the k with the highest silhouette score."""
    return max(scores, key=scores.__getitem__)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster(
    X: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    random_state: int = 42,
) -> np.ndarray:
    """Fit and return integer cluster labels for X."""
    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters, random_state=random_state,
            n_init=20, max_iter=500,
        )
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return model.fit_predict(X)


def pca_reduce(
    feat_df: pd.DataFrame,
    n_components: int = 30,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """PCA-reduce a feature DataFrame before clustering."""
    n = min(n_components, feat_df.shape[1], feat_df.shape[0] - 1)
    pca = PCA(n_components=n, random_state=random_state)
    X = pca.fit_transform(feat_df.values)
    return X, pca


# ---------------------------------------------------------------------------
# Cluster label alignment across time (Hungarian matching)
# ---------------------------------------------------------------------------

def align_labels(
    prev_labels: pd.Series,
    curr_labels: pd.Series,
) -> pd.Series:
    """
    Remap curr_labels so that cluster IDs are maximally consistent
    with prev_labels.  Uses the Hungarian algorithm on the overlap matrix.
    """
    common = prev_labels.index.intersection(curr_labels.index)
    if len(common) == 0:
        return curr_labels

    n = int(max(prev_labels.max(), curr_labels.max())) + 1
    overlap = np.zeros((n, n), dtype=int)
    for p, c in zip(prev_labels.loc[common], curr_labels.loc[common]):
        overlap[int(p), int(c)] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap)
    remap = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}

    return curr_labels.map(lambda x: remap.get(int(x), int(x)))


# ---------------------------------------------------------------------------
# Rolling temporal clustering
# ---------------------------------------------------------------------------

def run_rolling_clustering(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    fundamentals: Optional[pd.DataFrame],
    window_weeks: int = 52,
    step_weeks: int = 4,
    n_clusters: int = 25,
    n_pca_dims: int = 30,
    n_pca_factors: int = 3,
) -> pd.DataFrame:
    """
    Re-cluster the universe at every `step_weeks` step using a trailing
    `window_weeks` return window.

    Returns a DataFrame  (dates × tickers)  of cluster IDs.
    Cluster labels are kept stable across periods via Hungarian matching.
    """
    from src.features import build_feature_matrix, normalize_features

    all_dates = returns.index
    start_idx = window_weeks
    step_dates = all_dates[start_idx::step_weeks]

    assignments: dict[pd.Timestamp, pd.Series] = {}
    prev_labels: Optional[pd.Series] = None

    for date in step_dates:
        logger.info(f"Clustering snapshot: {date.date()}")
        window_ret = returns.loc[:date].tail(window_weeks)
        if window_ret.shape[0] < window_weeks // 2:
            continue

        feat = build_feature_matrix(
            window_ret, market_returns, fundamentals, window_weeks, n_pca_factors
        )
        norm = normalize_features(feat)
        if len(norm) < n_clusters * 2:
            logger.warning(f"  Too few stocks ({len(norm)}), skipping {date.date()}")
            continue

        X, _ = pca_reduce(norm, n_components=n_pca_dims)
        labels_arr = cluster(X, n_clusters=n_clusters)
        labels = pd.Series(labels_arr, index=norm.index, name=date)

        if prev_labels is not None:
            labels = align_labels(prev_labels, labels)

        assignments[date] = labels
        prev_labels = labels

    if not assignments:
        raise RuntimeError("No clustering snapshots produced — check window_weeks vs data length.")

    result = pd.DataFrame(assignments).T
    result.index.name = "date"
    return result


# ---------------------------------------------------------------------------
# Peer finding
# ---------------------------------------------------------------------------

def find_peers(
    ticker: str,
    cluster_assignments: pd.DataFrame,
    returns: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
    n_peers: int = 25,
    corr_window_weeks: int = 52,
) -> pd.DataFrame:
    """
    Return the `n_peers` closest cluster-mates for `ticker` as of `as_of_date`,
    ranked by trailing return correlation.

    Columns: ticker, cluster, return_correlation, total_return_52w, ann_vol.
    """
    if as_of_date is None:
        as_of_date = cluster_assignments.index[-1]

    # Find nearest snapshot date
    avail = cluster_assignments.index[cluster_assignments.index <= as_of_date]
    if avail.empty:
        raise ValueError(f"No cluster snapshot on or before {as_of_date}")
    snap_date = avail[-1]
    snap = cluster_assignments.loc[snap_date]

    if ticker not in snap.index:
        raise ValueError(f"{ticker!r} not in cluster snapshot for {snap_date.date()}")

    cluster_id = int(snap[ticker])
    peers = snap[snap == cluster_id].index.difference([ticker]).tolist()

    # Rank by return correlation
    window_ret = returns.loc[:as_of_date].tail(corr_window_weeks)
    anchor = window_ret.get(ticker)

    rows = []
    for p in peers:
        peer_ret = window_ret.get(p)
        corr = float(anchor.corr(peer_ret)) if (anchor is not None and peer_ret is not None) else float("nan")
        total_ret = float((1 + peer_ret.dropna()).prod() - 1) if peer_ret is not None else float("nan")
        ann_vol = float(peer_ret.dropna().std() * np.sqrt(52)) if peer_ret is not None else float("nan")
        rows.append({
            "ticker": p,
            "cluster": cluster_id,
            "return_correlation": corr,
            "total_return_52w": total_ret,
            "ann_vol": ann_vol,
        })

    peer_df = (
        pd.DataFrame(rows)
        .sort_values("return_correlation", ascending=False)
        .reset_index(drop=True)
    )
    return peer_df.head(n_peers)


# ---------------------------------------------------------------------------
# Peer history / cluster drift
# ---------------------------------------------------------------------------

def get_peer_history(
    ticker: str,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Track how a stock's cluster and peer group change across every snapshot.

    Returns DataFrame indexed by date with columns:
      cluster, cluster_size, new_peer_count, lost_peer_count, peer_churn_pct
    """
    if ticker not in cluster_assignments.columns:
        raise ValueError(f"{ticker!r} not found in cluster assignment matrix")

    rows = []
    prev_peers: set[str] = set()

    for date, snap in cluster_assignments.iterrows():
        if ticker not in snap.index or pd.isna(snap[ticker]):
            continue

        cid = int(snap[ticker])
        peers = set(snap[snap == cid].index.tolist()) - {ticker}

        new_count = len(peers - prev_peers) if prev_peers else 0
        lost_count = len(prev_peers - peers) if prev_peers else 0
        churn = (new_count + lost_count) / max(len(peers), 1)

        rows.append({
            "date": date,
            "cluster": cid,
            "cluster_size": len(peers) + 1,
            "new_peer_count": new_count,
            "lost_peer_count": lost_count,
            "peer_churn_pct": churn,
        })
        prev_peers = peers

    return pd.DataFrame(rows).set_index("date")


def cluster_composition(
    cluster_id: int,
    cluster_assignments: pd.DataFrame,
    fundamentals: Optional[pd.DataFrame],
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Return all members of a cluster at a given date with their sector/name.
    """
    if as_of_date is None:
        as_of_date = cluster_assignments.index[-1]
    avail = cluster_assignments.index[cluster_assignments.index <= as_of_date]
    snap = cluster_assignments.loc[avail[-1]]
    members = snap[snap == cluster_id].index.tolist()

    if fundamentals is None:
        return pd.DataFrame({"ticker": members})

    cols = [c for c in ["short_name", "gics_sector_name", "gics_industry_name"] if c in fundamentals.columns]
    info = fundamentals.loc[fundamentals.index.isin(members), cols].reset_index()
    info.columns = ["ticker"] + cols
    return info
