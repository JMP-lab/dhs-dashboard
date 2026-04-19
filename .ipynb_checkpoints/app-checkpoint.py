import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st


st.set_page_config(
    page_title="Discourse Health Score (DHS)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
    .metric-card {
        background: rgba(17, 25, 40, 0.88);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 0.7rem;
    }
    .small-note {color: #9aa4b2; font-size: 0.9rem;}
    .status-chip {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data model ----------
REQUIRED_COLUMNS = [
    "timestamp",
    "event_name",
    "user_id",
    "post_id",
    "reply_to_post_id",
    "reply_depth",
    "comment_sentiment",
    "reply_sentiment",
    "topic_similarity",
    "topic_entropy",
]

OPTIONAL_COLUMNS = [
    "platform",
    "source_type",
    "is_elite_source",
    "cluster_id",
]


@dataclass
class DHSMetrics:
    perspective_divergence: float
    interaction_structure: float
    temporal_instability: float
    dhs_score: float
    centralization: float
    bridging: float
    avg_reply_depth: float
    joint_valence_divergence: float
    topic_fragmentation: float
    perspective_instability: float
    participation_volatility: float
    lag1_autocorr: float
    risk_level: str
    status_label: str
    intervention_window: str


# ---------- Generic helpers ----------
def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if len(s) == 0:
        return pd.Series(dtype=float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def zsafe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    sd = s.std()
    mean = s.mean()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / sd


def infer_status(score: float) -> str:
    if score >= 0.72:
        return "Coordinated"
    if score >= 0.52:
        return "Transitioning"
    if score >= 0.32:
        return "Fragmenting"
    return "Polarized"


def infer_risk(score: float) -> str:
    if score >= 0.72:
        return "Low"
    if score >= 0.52:
        return "Moderate"
    if score >= 0.32:
        return "High"
    return "Critical"


def infer_intervention_window(score: float) -> str:
    if score >= 0.72:
        return "Monitor"
    if score >= 0.52:
        return "Open"
    if score >= 0.32:
        return "Narrow"
    return "Closing"


def safe_autocorr(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4 or s.nunique() <= 1:
        return 0.0
    return float(np.clip(s.autocorr(lag=1), -1, 1))


# ---------- Demo data ----------
def generate_demo_data(n_hours: int = 48, event_name: str = "Oil Shock") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-04-10 08:00", periods=n_hours, freq="h")
    records = []
    users = [f"u_{i:03d}" for i in range(1, 220)]
    elite_users = {"JDVance", "PeterThiel", "realDonaldTrump", "jimcramer", "WSJ", "Polymarket"}
    clusters = ["Cluster 1", "Cluster 2", "Cluster 3", "Bridging"]

    for i, ts in enumerate(timestamps):
        base_comments = int(70 + 20 * math.sin(i / 4) + rng.normal(0, 8))
        shock = 0
        if 18 <= i <= 32:
            shock = int(55 + 25 * math.sin((i - 18) / 3))
        comments = max(25, base_comments + shock)

        for j in range(comments):
            user = rng.choice(users)
            post_id = f"p_{i:03d}_{j:03d}"
            is_elite = rng.random() < 0.05
            source_type = "elite" if is_elite else "public"
            if is_elite:
                user = rng.choice(sorted(list(elite_users)))

            reply_count = max(0, int(rng.poisson(1.8 + (shock / 90))))
            topic_similarity = bounded(0.72 - 0.012 * max(0, i - 18) + rng.normal(0, 0.08))
            topic_entropy = bounded(0.34 + 0.018 * max(0, i - 18) + rng.normal(0, 0.07))
            comment_sent = bounded(0.55 + rng.normal(0, 0.18), 0, 1) * 2 - 1

            records.append(
                {
                    "timestamp": ts,
                    "event_name": event_name,
                    "platform": "X" if rng.random() < 0.7 else "TikTok",
                    "user_id": user,
                    "post_id": post_id,
                    "reply_to_post_id": None,
                    "reply_depth": 0,
                    "comment_sentiment": comment_sent,
                    "reply_sentiment": np.nan,
                    "topic_similarity": topic_similarity,
                    "topic_entropy": topic_entropy,
                    "source_type": source_type,
                    "is_elite_source": int(is_elite),
                    "cluster_id": rng.choice(clusters, p=[0.35, 0.27, 0.24, 0.14]),
                }
            )

            for r in range(reply_count):
                replier = rng.choice(users)
                gap = rng.normal(0.12 + 0.02 * max(0, i - 20), 0.18)
                reply_sent = float(np.clip(comment_sent - gap, -1, 1))
                depth = int(np.clip(rng.poisson(1.6 + 0.03 * max(0, i - 18)), 1, 7))
                records.append(
                    {
                        "timestamp": ts + pd.Timedelta(minutes=int(rng.integers(1, 50))),
                        "event_name": event_name,
                        "platform": "X" if rng.random() < 0.7 else "TikTok",
                        "user_id": replier,
                        "post_id": f"r_{i:03d}_{j:03d}_{r:02d}",
                        "reply_to_post_id": post_id,
                        "reply_depth": depth,
                        "comment_sentiment": comment_sent,
                        "reply_sentiment": reply_sent,
                        "topic_similarity": bounded(topic_similarity - rng.normal(0.06, 0.07)),
                        "topic_entropy": bounded(topic_entropy + rng.normal(0.05, 0.05)),
                        "source_type": "public",
                        "is_elite_source": 0,
                        "cluster_id": rng.choice(clusters, p=[0.25, 0.25, 0.35, 0.15]),
                    }
                )

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


# ---------- EPL loader ----------
@st.cache_data(show_spinner=False)
def list_epl_video_ids(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        return []

    video_ids = []
    for f in folder.glob("*_comments_v2.*"):
        name = f.name
        for suffix in ["_comments_v2.csv", "_comments_v2.xlsx", "_comments_v2.xls"]:
            if name.endswith(suffix):
                video_ids.append(name.replace(suffix, ""))
                break
    return sorted(set(video_ids))


def read_table_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            except Exception:
                pass
        raise ValueError(f"Could not read CSV: {path.name}")
    return pd.read_excel(path)


def standardize_comments(df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    rename_map = {}
    for c in out.columns:
        cl = c.lower()
        if cl in {"video_id", "videoid"}:
            rename_map[c] = "video_id"
        elif cl in {"comment_id", "commentid", "node_id", "id"}:
            rename_map[c] = "node_id"
        elif cl in {"comment", "text", "comment_text", "body", "message"}:
            rename_map[c] = "text"
        elif cl in {"author", "user", "username", "author_name", "channel"}:
            rename_map[c] = "user"
        elif cl in {"time", "timestamp", "published_at", "publishedat", "date"}:
            rename_map[c] = "timestamp"

    out = out.rename(columns=rename_map)

    if "video_id" not in out.columns:
        out["video_id"] = video_id
    if "node_id" not in out.columns:
        out["node_id"] = [f"{video_id}_comment_{i}" for i in range(len(out))]
    if "text" not in out.columns:
        out["text"] = ""
    if "user" not in out.columns:
        out["user"] = "unknown_user"
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT

    out["parent_id"] = pd.NA
    out["type"] = "comment"

    return out[["video_id", "node_id", "parent_id", "text", "user", "timestamp", "type"]]


def standardize_replies(df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    rename_map = {}
    for c in out.columns:
        cl = c.lower()
        if cl in {"video_id", "videoid"}:
            rename_map[c] = "video_id"
        elif cl in {"comment_id", "parent_id", "reply_to_comment_id", "reply_to", "parent"}:
            rename_map[c] = "parent_id"
        elif cl in {"reply_id", "node_id", "id"}:
            rename_map[c] = "node_id"
        elif cl in {"reply", "text", "reply_text", "body", "message"}:
            rename_map[c] = "text"
        elif cl in {"author", "user", "username", "author_name", "channel"}:
            rename_map[c] = "user"
        elif cl in {"time", "timestamp", "published_at", "publishedat", "date"}:
            rename_map[c] = "timestamp"

    out = out.rename(columns=rename_map)

    if "video_id" not in out.columns:
        out["video_id"] = video_id
    if "parent_id" not in out.columns:
        out["parent_id"] = pd.NA
    if "node_id" not in out.columns:
        out["node_id"] = [f"{video_id}_reply_{i}" for i in range(len(out))]
    if "text" not in out.columns:
        out["text"] = ""
    if "user" not in out.columns:
        out["user"] = "unknown_user"
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT

    out["type"] = "reply"

    return out[["video_id", "node_id", "parent_id", "text", "user", "timestamp", "type"]]


@st.cache_data(show_spinner=True)
def load_single_epl_video(folder_path: str, video_id: str) -> pd.DataFrame:
    folder = Path(folder_path)

    comment_file = None
    for ext in [".csv", ".xlsx", ".xls"]:
        p = folder / f"{video_id}_comments_v2{ext}"
        if p.exists():
            comment_file = p
            break

    if comment_file is None:
        return pd.DataFrame()

    df_comments = standardize_comments(read_table_any(comment_file), video_id)

    reply_file = None
    for ext in [".csv", ".xlsx", ".xls"]:
        p = folder / f"{video_id}_replies_v2{ext}"
        if p.exists():
            reply_file = p
            break

    if reply_file is not None:
        df_replies = standardize_replies(read_table_any(reply_file), video_id)
        out = pd.concat([df_comments, df_replies], ignore_index=True)
    else:
        out = df_comments.copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["node_id", "user"]).reset_index(drop=True)

    return out


def compute_reply_depth(df: pd.DataFrame) -> pd.Series:
    parent_lookup = dict(zip(df["node_id"], df["parent_id"]))
    memo = {}

    def depth(node_id):
        if node_id in memo:
            return memo[node_id]
        parent = parent_lookup.get(node_id)
        if pd.isna(parent) or parent is None:
            memo[node_id] = 0
            return 0
        if parent == node_id:
            memo[node_id] = 1
            return 1
        memo[node_id] = 1 + depth(parent)
        return memo[node_id]

    return df["node_id"].map(depth).fillna(0).astype(int)


def convert_epl_to_dhs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = {"video_id", "node_id", "parent_id", "user", "timestamp", "type"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing EPL columns after loading: {sorted(missing)}")

    out["post_id"] = out["node_id"].astype(str)
    out["reply_to_post_id"] = out["parent_id"]
    out["event_name"] = out["video_id"].astype(str)
    out["user_id"] = out["user"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["reply_depth"] = compute_reply_depth(out)

    is_reply = out["type"].eq("reply")
    out["comment_sentiment"] = np.where(is_reply, 0.0, 0.0)
    out["reply_sentiment"] = np.where(is_reply, 0.0, np.nan)
    out["topic_similarity"] = 0.5
    out["topic_entropy"] = 0.5

    out["platform"] = "YouTube"
    out["source_type"] = "public"
    out["is_elite_source"] = 0
    out["cluster_id"] = "Cluster 1"

    return out[
        [
            "timestamp",
            "event_name",
            "video_id",
            "user_id",
            "post_id",
            "reply_to_post_id",
            "reply_depth",
            "comment_sentiment",
            "reply_sentiment",
            "topic_similarity",
            "topic_entropy",
            "platform",
            "source_type",
            "is_elite_source",
            "cluster_id",
        ]
    ]


# ---------- Prep ----------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    for col in [
        "reply_depth",
        "comment_sentiment",
        "reply_sentiment",
        "topic_similarity",
        "topic_entropy",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "platform" not in data.columns:
        data["platform"] = "Unknown"
    if "source_type" not in data.columns:
        data["source_type"] = "public"
    if "is_elite_source" not in data.columns:
        data["is_elite_source"] = 0
    if "cluster_id" not in data.columns:
        data["cluster_id"] = "Cluster 1"

    data = data.dropna(subset=["timestamp", "event_name", "user_id", "post_id"]).copy()
    data["joint_valence_gap"] = (data["reply_sentiment"] - data["comment_sentiment"]).abs()
    data["is_reply"] = data["reply_to_post_id"].notna().astype(int)
    data["hour"] = data["timestamp"].dt.floor("h")
    return data


# ---------- Network and DHS ----------
def build_network(df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for _, row in df.iterrows():
        user = str(row["user_id"])
        g.add_node(user, source_type=row.get("source_type", "public"))

    parent_map = df.set_index("post_id")["user_id"].to_dict()
    reply_rows = df[df["reply_to_post_id"].notna()]
    for _, row in reply_rows.iterrows():
        src = str(row["user_id"])
        tgt = str(parent_map.get(row["reply_to_post_id"], f"post::{row['reply_to_post_id']}"))
        if src == tgt:
            continue
        if g.has_edge(src, tgt):
            g[src][tgt]["weight"] += 1
        else:
            g.add_edge(src, tgt, weight=1)
    return g


def compute_network_scores(g: nx.Graph) -> Tuple[float, float, float, dict, dict, dict]:
    if g.number_of_nodes() == 0:
        return 0.0, 0.0, 0.0, {}, {}, {}

    degree = nx.degree_centrality(g)
    betweenness = nx.betweenness_centrality(g, normalized=True)

    try:
        eigenvector = nx.eigenvector_centrality_numpy(g)
    except Exception:
        eigenvector = {n: 0.0 for n in g.nodes}

    centralization = float(np.mean(list(eigenvector.values()))) if eigenvector else 0.0
    bridging = float(np.mean(list(betweenness.values()))) if betweenness else 0.0

    try:
        community_map = nx.community.louvain_communities(g, seed=42)
        mod = nx.community.modularity(g, community_map)
    except Exception:
        mod = 0.0

    return bounded(centralization * 4), bounded(bridging * 8), bounded(mod), degree, betweenness, eigenvector


def compute_hourly(df: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        df.groupby("hour")
        .agg(
            comments=("post_id", "count"),
            replies=("is_reply", "sum"),
            joint_valence_gap=("joint_valence_gap", "mean"),
            topic_entropy=("topic_entropy", "mean"),
            topic_similarity=("topic_similarity", "mean"),
            avg_reply_depth=("reply_depth", "mean"),
            elite_share=("is_elite_source", "mean"),
        )
        .reset_index()
    )
    hourly["topic_fragmentation"] = 1 - hourly["topic_similarity"].fillna(hourly["topic_similarity"].median())
    return hourly


def compute_dhs(df: pd.DataFrame) -> Tuple[DHSMetrics, pd.DataFrame, nx.Graph, dict, dict, dict]:
    g = build_network(df)
    centralization, bridging, modularity, degree, betweenness, eigenvector = compute_network_scores(g)
    hourly = compute_hourly(df)

    joint_div = float(df["joint_valence_gap"].mean(skipna=True))
    topic_frag = float(hourly["topic_fragmentation"].mean(skipna=True))
    avg_reply_depth = float(df.loc[df["is_reply"] == 1, "reply_depth"].mean(skipna=True) or 0.0)

    perspective_instability = float(hourly["joint_valence_gap"].std(skipna=True) or 0.0)
    participation_volatility = float(hourly["replies"].std(skipna=True) or 0.0)
    lag1 = safe_autocorr(hourly["replies"])

    joint_div_n = bounded(joint_div / 1.2)
    topic_frag_n = bounded(topic_frag)
    reply_depth_n = bounded(avg_reply_depth / 6.0)
    perspective_instability_n = bounded(perspective_instability / 0.5)
    participation_volatility_n = bounded(participation_volatility / max(20.0, hourly["replies"].mean() + 1))
    lag1_n = bounded((lag1 + 1) / 2)

    perspective_divergence = bounded(0.55 * joint_div_n + 0.45 * topic_frag_n)
    interaction_structure = bounded(
        0.45 * centralization + 0.25 * (1 - bridging) + 0.20 * modularity + 0.10 * reply_depth_n
    )
    temporal_instability = bounded(
        0.40 * perspective_instability_n + 0.35 * participation_volatility_n + 0.25 * lag1_n
    )

    risk_pressure = bounded(
        0.38 * perspective_divergence + 0.32 * interaction_structure + 0.30 * temporal_instability
    )
    dhs_score = bounded(1 - risk_pressure)

    metrics = DHSMetrics(
        perspective_divergence=perspective_divergence,
        interaction_structure=interaction_structure,
        temporal_instability=temporal_instability,
        dhs_score=dhs_score,
        centralization=centralization,
        bridging=bridging,
        avg_reply_depth=avg_reply_depth,
        joint_valence_divergence=joint_div,
        topic_fragmentation=topic_frag,
        perspective_instability=perspective_instability,
        participation_volatility=participation_volatility,
        lag1_autocorr=lag1,
        risk_level=infer_risk(dhs_score),
        status_label=infer_status(dhs_score),
        intervention_window=infer_intervention_window(dhs_score),
    )
    return metrics, hourly, g, degree, betweenness, eigenvector


# ---------- AMI ----------
def compute_cross_cluster_hourly(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()

    if "cluster_id" not in tmp.columns:
        return pd.DataFrame({"hour": sorted(tmp["hour"].dropna().unique()), "cross_cluster": 0.0})

    parent_cluster = tmp.set_index("post_id")["cluster_id"].to_dict()
    tmp["parent_cluster"] = tmp["reply_to_post_id"].map(parent_cluster)

    tmp["cross_cluster"] = (
        tmp["reply_to_post_id"].notna()
        & tmp["cluster_id"].notna()
        & tmp["parent_cluster"].notna()
        & (tmp["cluster_id"] != tmp["parent_cluster"])
    ).astype(int)

    return tmp.groupby("hour", as_index=False)["cross_cluster"].mean()


def compute_new_user_hourly(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    first_seen = tmp.groupby("user_id")["hour"].min().rename("first_hour").reset_index()
    tmp = tmp.merge(first_seen, on="user_id", how="left")
    tmp["new_user_entry"] = (tmp["hour"] == tmp["first_hour"]).astype(int)
    return tmp.groupby("hour", as_index=False)["new_user_entry"].mean()


def compute_ami_hourly(df: pd.DataFrame, hourly: pd.DataFrame, bridging_value: float) -> pd.DataFrame:
    temp = hourly.copy().sort_values("hour").reset_index(drop=True)

    if temp.empty:
        return temp

    temp["replies_lag1"] = temp["replies"].shift(1)
    temp["comments_lag1"] = temp["comments"].shift(1)
    temp["jv_lag1"] = temp["joint_valence_gap"].shift(1)
    temp["frag_lag1"] = temp["topic_fragmentation"].shift(1)
    temp["elite_lag1"] = temp["elite_share"].shift(1)

    temp["reply_velocity"] = temp["replies"].diff()
    temp["reply_acceleration"] = temp["reply_velocity"].diff()

    temp["expected_replies"] = (
        0.55 * temp["replies_lag1"].fillna(temp["replies"].median())
        + 0.20 * temp["comments_lag1"].fillna(temp["comments"].median())
        + 0.10 * temp["elite_lag1"].fillna(0) * temp["comments"].median()
        + 0.15 * (1 - temp["frag_lag1"].fillna(temp["topic_fragmentation"].median())) * temp["replies"].median()
    )

    temp["excess_attention"] = temp["replies"] - temp["expected_replies"]

    cross_hourly = compute_cross_cluster_hourly(df)
    entry_hourly = compute_new_user_hourly(df)

    temp = temp.merge(cross_hourly, on="hour", how="left")
    temp = temp.merge(entry_hourly, on="hour", how="left")

    temp["cross_cluster"] = temp["cross_cluster"].fillna(0.0)
    temp["new_user_entry"] = temp["new_user_entry"].fillna(0.0)
    temp["reply_acceleration"] = temp["reply_acceleration"].fillna(0.0)

    temp["z_excess_attention"] = zsafe(temp["excess_attention"])
    temp["z_reply_acceleration"] = zsafe(temp["reply_acceleration"])
    temp["z_cross_cluster"] = zsafe(temp["cross_cluster"])
    temp["z_new_user_entry"] = zsafe(temp["new_user_entry"])

    temp["ami_score"] = (
        0.35 * temp["z_excess_attention"].fillna(0)
        + 0.25 * temp["z_reply_acceleration"].fillna(0)
        + 0.20 * temp["z_cross_cluster"].fillna(0)
        + 0.20 * temp["z_new_user_entry"].fillna(0)
    )

    temp["bridging_proxy"] = bridging_value

    temp["rolling_dhs"] = 1 - minmax(
        0.38 * minmax(temp["joint_valence_gap"].fillna(0))
        + 0.32 * minmax(temp["topic_fragmentation"].fillna(0))
        + 0.30 * minmax(temp["replies"].fillna(0))
    )

    spike_cut = temp["ami_score"].quantile(0.90) if len(temp) >= 10 else temp["ami_score"].max()
    temp["ami_spike"] = (temp["ami_score"] >= spike_cut).astype(int)

    return temp


def run_simple_causal_test(ami_hourly: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    temp = ami_hourly.copy().sort_values("hour").reset_index(drop=True)

    if temp.empty:
        return pd.DataFrame()

    temp["future_dhs"] = temp["rolling_dhs"].shift(-horizon)
    temp["ami_x_bridging"] = temp["ami_score"] * (1 - temp["bridging_proxy"])
    temp["ami_x_fragmentation"] = temp["ami_score"] * temp["topic_fragmentation"].fillna(0)

    needed = [
        "future_dhs",
        "ami_score",
        "rolling_dhs",
        "bridging_proxy",
        "topic_fragmentation",
        "ami_x_bridging",
        "ami_x_fragmentation",
    ]
    model_df = temp[needed].dropna()

    if len(model_df) < 10:
        return pd.DataFrame()

    X = model_df[
        ["ami_score", "rolling_dhs", "bridging_proxy", "topic_fragmentation", "ami_x_bridging", "ami_x_fragmentation"]
    ]
    X = sm.add_constant(X)
    y = model_df["future_dhs"]

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return pd.DataFrame()

    conf = model.conf_int()
    out = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "t": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
        }
    )
    return out


def build_event_study_frame(ami_hourly: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    temp = ami_hourly.copy().sort_values("hour").reset_index(drop=True)

    if temp.empty or "ami_spike" not in temp.columns:
        return pd.DataFrame()

    spike_idx = temp.index[temp["ami_spike"] == 1].tolist()
    if not spike_idx:
        return pd.DataFrame()

    rows = []
    bridging_median = temp["bridging_proxy"].median()
    frag_median = temp["topic_fragmentation"].median()

    for idx in spike_idx:
        base_bridging = temp.loc[idx, "bridging_proxy"]
        base_frag = temp.loc[idx, "topic_fragmentation"]

        for rel in range(-window, window + 1):
            j = idx + rel
            if 0 <= j < len(temp):
                rows.append(
                    {
                        "event_id": idx,
                        "relative_hour": rel,
                        "rolling_dhs": temp.loc[j, "rolling_dhs"],
                        "group_bridging": "High bridging" if base_bridging >= bridging_median else "Low bridging",
                        "group_fragmentation": "High fragmentation" if base_frag >= frag_median else "Low fragmentation",
                    }
                )

    return pd.DataFrame(rows)


# ---------- Figures ----------
def build_network_figure(
    g: nx.Graph,
    degree: dict,
    betweenness: dict,
    eigenvector: dict,
    top_n: int = 80,
) -> go.Figure:
    if g.number_of_nodes() == 0:
        return go.Figure()

    ranked = sorted(g.nodes(), key=lambda n: degree.get(n, 0) + betweenness.get(n, 0), reverse=True)[:top_n]
    sg = g.subgraph(ranked).copy()
    pos = nx.spring_layout(sg, seed=42, k=0.45)

    edge_x, edge_y = [], []
    for u, v in sg.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="rgba(160,170,190,0.25)"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
    for n in sg.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_size.append(10 + 90 * eigenvector.get(n, 0))
        node_color.append(betweenness.get(n, 0))
        node_text.append(
            f"User: {n}<br>Degree: {degree.get(n, 0):.3f}<br>Betweenness: {betweenness.get(n, 0):.3f}<br>Eigenvector: {eigenvector.get(n, 0):.3f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Plasma",
            color=node_color,
            size=node_size,
            colorbar=dict(title="Betweenness"),
            line=dict(width=1, color="rgba(255,255,255,0.6)"),
            opacity=0.92,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=340,
    )
    return fig


def gauge_figure(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, 0.32], "color": "#8B1E3F"},
                    {"range": [0.32, 0.52], "color": "#C97A00"},
                    {"range": [0.52, 0.72], "color": "#8B8A12"},
                    {"range": [0.72, 1], "color": "#0B8F55"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "value": score},
            },
        )
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), height=240)
    return fig


def status_color(status: str) -> str:
    return {
        "Coordinated": "#0B8F55",
        "Transitioning": "#B58B00",
        "Fragmenting": "#D96C00",
        "Polarized": "#B42318",
    }.get(status, "#64748B")


def risk_badges(metrics: DHSMetrics) -> list[str]:
    badges = []
    if metrics.joint_valence_divergence > 0.35:
        badges.append("Rising joint valence divergence")
    if metrics.topic_fragmentation > 0.35:
        badges.append("Increasing topic fragmentation")
    if metrics.bridging < 0.12:
        badges.append("Cross-cluster bridging is weakening")
    if metrics.participation_volatility > 40:
        badges.append("Participation volatility is elevated")
    if metrics.lag1_autocorr > 0.45:
        badges.append("Persistence suggests critical slowing down")
    return badges[:5]


def intervention_suggestions(metrics: DHSMetrics) -> list[str]:
    actions = []
    if metrics.bridging < 0.12:
        actions.append("Introduce bridging prompts or cross-cluster summaries.")
    if metrics.centralization > 0.22:
        actions.append("Reduce over-concentration by broadening exposure beyond dominant actors.")
    if metrics.joint_valence_divergence > 0.35:
        actions.append("Inject clarifying or context-preserving content before divergence hardens.")
    if metrics.topic_fragmentation > 0.35:
        actions.append("Cluster topics and surface shared anchors to reduce semantic drift.")
    if metrics.participation_volatility > 40:
        actions.append("Throttle burst amplification and monitor for cascading replies.")
    return actions[:4] or ["System is relatively stable; continue monitoring."]


# ---------- Sidebar ----------
st.sidebar.title("DHS Control Panel")

mode = st.sidebar.radio(
    "Data source",
    ["Demo data", "EPL threaded"],
    index=0,
)

default_epl_path = r"C:\Users\JMPark\Documents\dhs_dashboard\data\threaded\epl"
epl_folder = st.sidebar.text_input("EPL folder path", value=default_epl_path)

if mode == "Demo data":
    uploaded_df = generate_demo_data()
    selected_event = "Demo"
else:
    video_ids = list_epl_video_ids(epl_folder)

    if not video_ids:
        st.sidebar.error("No EPL videos found in folder.")
        st.stop()

    selected_video = st.sidebar.selectbox("Video / Event", video_ids, index=0)

    with st.spinner(f"Loading {selected_video}..."):
        epl_raw = load_single_epl_video(epl_folder, selected_video)

    if epl_raw.empty:
        st.error("Could not load selected EPL video.")
        st.stop()

    uploaded_df = convert_epl_to_dhs(epl_raw)
    selected_event = selected_video

st.sidebar.markdown("---")
with st.sidebar.expander("Expected columns", expanded=False):
    st.write(REQUIRED_COLUMNS)
    st.caption("Optional: platform, source_type, is_elite_source, cluster_id")

# ---------- Prepare ----------
df = prepare_data(uploaded_df)

if df.empty:
    st.error("No usable records after preprocessing.")
    st.stop()

is_placeholder_epl = (
    mode == "EPL threaded"
    and df["comment_sentiment"].fillna(0).eq(0).all()
    and df["reply_sentiment"].fillna(0).eq(0).all()
    and df["topic_similarity"].fillna(0.5).eq(0.5).all()
    and df["topic_entropy"].fillna(0.5).eq(0.5).all()
)

if "video_id" in df.columns and mode == "EPL threaded":
    filtered = df.copy()
else:
    event_names = sorted(df["event_name"].dropna().astype(str).unique().tolist())
    selected_event = st.sidebar.selectbox("Event", event_names, index=0)
    filtered = df[df["event_name"] == selected_event].copy()

platforms = sorted(filtered["platform"].dropna().astype(str).unique().tolist())
selected_platforms = st.sidebar.multiselect("Platform", platforms, default=platforms)

if selected_platforms:
    filtered = filtered[filtered["platform"].isin(selected_platforms)]
else:
    st.warning("No platform selected.")
    st.stop()

if filtered.empty or filtered["timestamp"].dropna().empty:
    st.error("No records available for the selected filters.")
    st.stop()

start_ts = filtered["timestamp"].min()
end_ts = filtered["timestamp"].max()

window = st.sidebar.slider(
    "Time window",
    min_value=start_ts.to_pydatetime(),
    max_value=end_ts.to_pydatetime(),
    value=(start_ts.to_pydatetime(), end_ts.to_pydatetime()),
    format="YYYY-MM-DD HH:mm",
)

filtered = filtered[
    (filtered["timestamp"] >= pd.Timestamp(window[0])) &
    (filtered["timestamp"] <= pd.Timestamp(window[1]))
]

if filtered.empty:
    st.error("No records available for the selected filters.")
    st.stop()

metrics, hourly, graph, degree, betweenness, eigenvector = compute_dhs(filtered)
ami_hourly = compute_ami_hourly(filtered, hourly, metrics.bridging)
causal_results = run_simple_causal_test(ami_hourly, horizon=1)
event_df = build_event_study_frame(ami_hourly, window=6)

# ---------- Header ----------
left, right = st.columns([3.2, 1.4])
with left:
    st.title("Discourse Health Score (DHS)")
    st.caption("Real-time monitoring of discourse stability, early warning signals, intervention timing, and AI-mediated amplification.")
with right:
    st.markdown(
        f"<div class='metric-card'><div class='small-note'>Selected event</div>"
        f"<div style='font-size:1.25rem;font-weight:700'>{selected_event}</div>"
        f"<div class='small-note'>{window[0].strftime('%b %d, %Y %H:%M')} → {window[1].strftime('%b %d, %Y %H:%M')}</div></div>",
        unsafe_allow_html=True,
    )

# ---------- AMI block ----------
st.markdown("## AI-Mediated Amplification")

if not ami_hourly.empty:
    latest_ami = ami_hourly.iloc[-1]
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("AMI score", f"{latest_ami['ami_score']:.2f}")
    a2.metric("Excess attention", f"{latest_ami['excess_attention']:.2f}")
    a3.metric("Velocity distortion", f"{latest_ami['reply_acceleration']:.2f}")
    a4.metric("Cross-cluster collision", f"{latest_ami['cross_cluster']:.2f}")
    a5.metric("Out-of-network entry", f"{latest_ami['new_user_entry']:.2f}")
    a6.metric("AMI spike", "Yes" if int(latest_ami["ami_spike"]) == 1 else "No")

    fig_ami = go.Figure()
    fig_ami.add_trace(
        go.Scatter(
            x=ami_hourly["hour"],
            y=ami_hourly["ami_score"],
            name="AMI",
            line=dict(width=3),
        )
    )
    fig_ami.add_hline(y=ami_hourly["ami_score"].quantile(0.90), line_dash="dash")
    fig_ami.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="AMI score",
    )
    st.plotly_chart(fig_ami, use_container_width=True, key="ami_score_chart")
else:
    st.info("AMI could not be computed for the current selection.")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Causal Tests", "Event Study"])

with tab1:
    st.subheader("Causal Tests")

    if causal_results.empty:
        st.info("Not enough hourly observations yet for causal testing.")
    else:
        st.dataframe(causal_results, use_container_width=True, hide_index=True)

        coef_df = causal_results[causal_results["term"] != "const"].copy()
        coef_fig = go.Figure()
        coef_fig.add_trace(
            go.Scatter(
                x=coef_df["coef"],
                y=coef_df["term"],
                mode="markers",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=coef_df["ci_high"] - coef_df["coef"],
                    arrayminus=coef_df["coef"] - coef_df["ci_low"],
                ),
            )
        )
        coef_fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Coefficient",
            yaxis_title="",
        )
        st.plotly_chart(coef_fig, use_container_width=True, key="causal_coef_chart")

with tab2:
    st.subheader("Event Study")

    if event_df.empty:
        st.info("No AMI spikes detected yet for event study.")
    else:
        view_choice = st.radio(
            "Split event study by:",
            ["Bridging", "Fragmentation"],
            horizontal=True,
            key="event_study_split",
        )

        if view_choice == "Bridging":
            plot_df = event_df.groupby(["relative_hour", "group_bridging"], as_index=False)["rolling_dhs"].mean()
            color_col = "group_bridging"
        else:
            plot_df = event_df.groupby(["relative_hour", "group_fragmentation"], as_index=False)["rolling_dhs"].mean()
            color_col = "group_fragmentation"

        fig_ev = px.line(
            plot_df,
            x="relative_hour",
            y="rolling_dhs",
            color=color_col,
            template="plotly_dark",
        )
        fig_ev.add_vline(x=0, line_dash="dash")
        fig_ev.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Hours relative to AMI spike",
            yaxis_title="Average DHS",
        )
        st.plotly_chart(fig_ev, use_container_width=True, key="event_study_chart")

# ---------- Top DHS row ----------
col1, col2, col3 = st.columns([1.4, 1.9, 1.8])

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("DHS Score")
    st.plotly_chart(gauge_figure(metrics.dhs_score), use_container_width=True, key="dhs_gauge")
    st.markdown(
        f"<span class='status-chip' style='background:{status_color(metrics.status_label)}22;color:{status_color(metrics.status_label)}'>"
        f"{metrics.status_label}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Risk level", metrics.risk_level)
    st.metric("Intervention window", metrics.intervention_window)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("System State Over Time")

    if is_placeholder_epl:
        temp = hourly.copy()
        fig_state = px.line(temp, x="hour", y="replies", template="plotly_dark")
        fig_state.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Replies",
        )
        st.plotly_chart(fig_state, use_container_width=True, key="system_state_replies_only")
        st.caption("EPL currently shows participation dynamics only. Semantic state metrics are not embedded yet.")
    else:
        temp = hourly.copy()
        temp["rolling_dhs"] = 1 - minmax(
            0.38 * minmax(temp["joint_valence_gap"].fillna(0))
            + 0.32 * minmax(temp["topic_fragmentation"].fillna(0))
            + 0.30 * minmax(temp["replies"].fillna(0))
        )
        fig_state = px.line(temp, x="hour", y="rolling_dhs", template="plotly_dark")
        fig_state.update_traces(line=dict(width=3, color="#E5E7EB"))
        fig_state.add_hrect(y0=0.72, y1=1.0, fillcolor="#0B8F55", opacity=0.18, line_width=0)
        fig_state.add_hrect(y0=0.52, y1=0.72, fillcolor="#B58B00", opacity=0.18, line_width=0)
        fig_state.add_hrect(y0=0.32, y1=0.52, fillcolor="#D96C00", opacity=0.18, line_width=0)
        fig_state.add_hrect(y0=0.0, y1=0.32, fillcolor="#B42318", opacity=0.18, line_width=0)
        fig_state.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="DHS")
        st.plotly_chart(fig_state, use_container_width=True, key="system_state_dhs")

    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Event Context")
    total_posts = int(filtered["post_id"].nunique())
    total_replies = int(filtered["is_reply"].sum())
    unique_users = int(filtered["user_id"].nunique())
    elite_share = float(filtered["is_elite_source"].mean()) if "is_elite_source" in filtered else 0.0
    context_df = pd.DataFrame(
        {
            "Metric": ["Platforms", "Total posts", "Total replies", "Unique users", "Elite share"],
            "Value": [
                ", ".join(selected_platforms),
                f"{total_posts:,}",
                f"{total_replies:,}",
                f"{unique_users:,}",
                f"{elite_share:.1%}",
            ],
        }
    )
    st.dataframe(context_df, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Middle row ----------
left, mid, right = st.columns(3)

with left:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("1. Perspective Divergence")
    st.caption("How meaning and perspectives evolve")

    if is_placeholder_epl:
        st.info("Perspective metrics are placeholders for EPL right now. Use this panel after sentiment/topic embedding is added.")
        fig_jv = go.Figure()
        fig_jv.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_jv, use_container_width=True, key="perspective_placeholder_chart")
    else:
        fig_jv = go.Figure()
        fig_jv.add_trace(
            go.Scatter(x=hourly["hour"], y=hourly["joint_valence_gap"], name="Joint valence gap", line=dict(width=3))
        )
        fig_jv.add_trace(
            go.Scatter(
                x=hourly["hour"],
                y=hourly["topic_fragmentation"],
                name="Topic fragmentation",
                line=dict(width=2, dash="dot"),
            )
        )
        fig_jv.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_jv, use_container_width=True, key="perspective_chart")

    m1, m2, m3 = st.columns(3)
    m1.metric("Divergence index", f"{metrics.perspective_divergence:.2f}")
    m2.metric("Sentiment gap", f"{metrics.joint_valence_divergence:.2f}")
    m3.metric("Topic fragmentation", f"{metrics.topic_fragmentation:.2f}")

    if is_placeholder_epl:
        st.info("Semantic interpretation is paused until real EPL sentiment/topic fields are embedded.")
    else:
        verb = "aligning" if metrics.perspective_divergence < 0.30 else "diverging"
        st.info(f"Perspectives are **{verb}** across comment–reply interactions.")

    st.markdown("</div>", unsafe_allow_html=True)

with mid:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("2. Interaction Structure")
    st.caption("How users connect and influence")
    st.plotly_chart(
        build_network_figure(graph, degree, betweenness, eigenvector),
        use_container_width=True,
        key="network_chart",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Centralization", f"{metrics.centralization:.2f}")
    m2.metric("Bridging", f"{metrics.bridging:.2f}")
    m3.metric("Avg reply depth", f"{metrics.avg_reply_depth:.1f}")
    m4.metric("Topology risk", f"{metrics.interaction_structure:.2f}")

    structure_text = "bridging" if metrics.bridging >= 0.14 else "concentrating"
    st.warning(f"Network is **{structure_text}**; coordination capacity is being reshaped.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("3. Temporal Instability")
    st.caption("How the system fluctuates over time")

    c1, c2 = st.columns(2)
    with c1:
        if is_placeholder_epl:
            fig_pi = px.line(hourly, x="hour", y="replies", template="plotly_dark")
            chart_key_left = "temp_instability_left_replies"
        else:
            fig_pi = px.line(hourly, x="hour", y="joint_valence_gap", template="plotly_dark")
            chart_key_left = "temp_instability_left_gap"
        fig_pi.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_pi, use_container_width=True, key=chart_key_left)
        st.metric("Perspective instability", f"{metrics.perspective_instability:.2f}")
        st.metric("Lag-1 autocorr", f"{metrics.lag1_autocorr:.2f}")

    with c2:
        fig_pv = px.line(hourly, x="hour", y="replies", template="plotly_dark")
        fig_pv.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_pv, use_container_width=True, key="temp_instability_right_replies")
        st.metric("Participation volatility", f"{metrics.participation_volatility:.1f}")
        st.metric("Instability risk", f"{metrics.temporal_instability:.2f}")

    if is_placeholder_epl:
        st.info("This EPL instability view emphasizes activity bursts and persistence first.")
    else:
        instability_text = "stabilizing" if metrics.temporal_instability < 0.35 else "destabilizing"
        st.error(f"System is **{instability_text}** as volatility and persistence rise.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Bottom row ----------
left, mid, right = st.columns([1.5, 0.9, 1.2])

with left:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Early Warning Signals")
    signals = risk_badges(metrics)
    if signals:
        for s in signals:
            st.write(f"⚠️ {s}")
    else:
        st.write("No major warnings detected.")
    st.markdown(f"**Overall system risk:** {metrics.risk_level}")
    st.markdown("</div>", unsafe_allow_html=True)

with mid:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Intervention Window")
    progress = {
        "Monitor": 0.2,
        "Open": 0.65,
        "Narrow": 0.9,
        "Closing": 0.98,
    }[metrics.intervention_window]
    st.progress(progress)
    st.metric("Window status", metrics.intervention_window)
    hint = {
        "Monitor": "System is healthy; continue observing.",
        "Open": "Useful moment for soft intervention.",
        "Narrow": "Urgent: coordination is weakening.",
        "Closing": "Late-stage instability; options are limited.",
    }[metrics.intervention_window]
    st.caption(hint)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("AI-Suggested Interventions")
    for action in intervention_suggestions(metrics):
        st.write(f"• {action}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
with st.expander("How DHS is computed", expanded=False):
    st.markdown(
        """
**DHS is a system-level stability score** derived from three dimensions:

- **Perspective Divergence**: joint valence gap + topic fragmentation
- **Interaction Structure**: centralization, bridging weakness, modularity, reply depth
- **Temporal Instability**: perspective instability, participation volatility, lag-1 autocorrelation

A higher DHS means the system is more coordinated and stable; a lower DHS indicates fragmentation risk.
        """
    )

with st.expander("How AMI is computed", expanded=False):
    st.markdown(
        """
**AMI (AI-Mediated Amplification)** is a prototype index built from:

- **Excess Attention**: observed replies minus expected replies
- **Velocity Distortion**: reply acceleration
- **Cross-Cluster Collision**: interaction across different clusters
- **Out-of-Network Entry**: share of newly entering users

Higher AMI suggests attention and participation are moving faster or farther than baseline diffusion would predict.
        """
    )