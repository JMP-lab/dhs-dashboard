import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG + TIGHT LAYOUT
# =========================================================
st.set_page_config(page_title="DHS / DSH Explorer", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.25rem !important;
        padding-bottom: 0.20rem !important;
        padding-left: 0.70rem !important;
        padding-right: 0.70rem !important;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        width: 255px !important;
        min-width: 255px !important;
        max-width: 255px !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.35rem !important;
    }

    .small-card {
        padding: 7px 9px;
        border-radius: 12px;
        border: 1px solid #d9d9d9;
        background: #fafafa;
        min-height: 78px;
    }

    .flow-card {
        padding: 8px 9px;
        border-radius: 13px;
        min-height: 112px;
        border-width: 1.2px !important;
        border-style: solid !important;
    }

    .flow-num {
        width: 22px;
        height: 22px;
        border-radius: 999px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.72rem;
        margin-bottom: 6px;
    }

    .flow-title {
        font-size: 0.78rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 1px;
    }

    .flow-sub {
        font-size: 0.64rem;
        color: #666;
        line-height: 1.0;
    }

    .flow-val {
        font-size: 0.86rem;
        font-weight: 800;
        margin-top: 12px;
        line-height: 1.15;
        color: #1f2937;
    }

    .element-container {
        margin-bottom: 0.28rem !important;
    }

    hr {
        margin-top: 0.55rem !important;
        margin-bottom: 0.55rem !important;
    }

    .stCaption {
        font-size: 0.72rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# APP SETTINGS
# =========================================================
APP_TITLE = "Discourse Dynamics Explorer"
APP_SUBTITLE = "Demo + EPL workflow for Volume, Meaning, Wave Type, Joint Valence, and Network Effect"

IMAGE_FILES = {
    "overall_flow": "overall_process_flow.png",
    "paths_1_4": "paths_1_4.png",
    "network_snapshot": "network_snapshot.png",
    "volume_vs_valence": "volume_vs_valence.png",
    "wave_vs_joint": "wave_vs_joint.png",
}


# =========================================================
# TABLE
# =========================================================
def build_operationalization_table() -> pd.DataFrame:
    rows = [
        {
            "Layer": "Surface / Level",
            "Variable": "Initial Volume",
            "Definition": "Starting attention level at the beginning of the observation window.",
            "Operationalization": "Mean or sum of comments + replies in the first k periods.",
            "Data source": "comments + replies files",
            "Unit": "count",
            "Interpretation": "Higher values indicate stronger initial visibility.",
        },
        {
            "Layer": "Surface / Level",
            "Variable": "Volume Change",
            "Definition": "How visible activity evolves over time.",
            "Operationalization": "Time series of comments + replies by day.",
            "Data source": "comments + replies files",
            "Unit": "daily count",
            "Interpretation": "Used to classify wave types.",
        },
        {
            "Layer": "Surface / Level",
            "Variable": "Wave Type",
            "Definition": "Temporal geometry of attention over time.",
            "Operationalization": "Classify by local maxima and directional change.",
            "Data source": "derived from daily volume",
            "Unit": "category",
            "Interpretation": "Represents visible momentum.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Comment Valence (VC)",
            "Definition": "Average sentiment of comments over time.",
            "Operationalization": "Daily mean sentiment score across comments.",
            "Data source": "comments file",
            "Unit": "sentiment score",
            "Interpretation": "Tracks directional tone in comments.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Reply Valence (VR)",
            "Definition": "Average sentiment of replies over time.",
            "Operationalization": "Daily mean sentiment score across replies.",
            "Data source": "replies file",
            "Unit": "sentiment score",
            "Interpretation": "Tracks directional tone in replies.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Sentiment Gap",
            "Definition": "Difference between comment and reply sentiment.",
            "Operationalization": "VC - VR by period.",
            "Data source": "derived from VC and VR",
            "Unit": "difference score",
            "Interpretation": "Larger gap indicates stronger misalignment.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Topic Concentration",
            "Definition": "How focused the discussion is.",
            "Operationalization": "1 - topic entropy or concentration proxy.",
            "Data source": "comments + replies text",
            "Unit": "index",
            "Interpretation": "Higher values mean discussion stays focused.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Meaning Fragmentation",
            "Definition": "Whether discussion splits into disconnected interpretations.",
            "Operationalization": "Semantic dispersion / topic entropy / cluster split.",
            "Data source": "comments + replies text",
            "Unit": "index",
            "Interpretation": "Higher values mean more scattered meanings.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Joint Valence",
            "Definition": "Co-evolutionary pattern between comment and reply valence.",
            "Operationalization": "Positive Resonance, Negative Entrenchment, Resilient Rebound, Divergence, Indeterminate Coupling.",
            "Data source": "derived from VC and VR time series",
            "Unit": "category",
            "Interpretation": "Represents hidden instability.",
        },
        {
            "Layer": "Network / Inflow",
            "Variable": "Degree",
            "Definition": "Local interaction intensity around users.",
            "Operationalization": "Average degree in user-reply network.",
            "Data source": "thread / reply network",
            "Unit": "network metric",
            "Interpretation": "Higher values mean stronger local interaction.",
        },
        {
            "Layer": "Network / Inflow",
            "Variable": "Bridging (Betweenness)",
            "Definition": "Cross-group brokerage connecting separate users.",
            "Operationalization": "Betweenness centrality.",
            "Data source": "thread / reply network",
            "Unit": "network metric",
            "Interpretation": "Higher values mean stronger brokerage.",
        },
        {
            "Layer": "Network / Inflow",
            "Variable": "Impact (Eigenvector)",
            "Definition": "Influence through ties to influential users.",
            "Operationalization": "Eigenvector centrality.",
            "Data source": "thread / reply network",
            "Unit": "network metric",
            "Interpretation": "Higher values mean stronger influence concentration.",
        },
        {
            "Layer": "Final Readout",
            "Variable": "Level",
            "Definition": "Visible strength of conversation.",
            "Operationalization": "Derived from volume trajectory and wave type.",
            "Data source": "volume + wave type",
            "Unit": "Low / Medium / High / Very High",
            "Interpretation": "Manager-facing visibility signal.",
        },
        {
            "Layer": "Final Readout",
            "Variable": "Risk",
            "Definition": "Hidden instability behind visible momentum.",
            "Operationalization": "Derived from joint valence and fragmentation.",
            "Data source": "meaning layer + joint valence",
            "Unit": "Low / Medium / High / Very High",
            "Interpretation": "Manager-facing hidden instability signal.",
        },
    ]
    return pd.DataFrame(rows)


# =========================================================
# DEMO DATA
# =========================================================
def make_demo_time_series(n_days: int = 14, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day = np.arange(1, n_days + 1)

    volume = np.array([5, 8, 14, 19, 27, 25, 18, 15, 17, 21, 18, 14, 11, 9], dtype=float)
    volume += rng.normal(0, 0.8, len(volume))

    vc = np.array([0.05, 0.08, 0.10, 0.14, 0.19, 0.16, 0.12, 0.10, 0.14, 0.18, 0.22, 0.26, 0.29, 0.31])
    vr = np.array([0.03, 0.05, 0.08, 0.12, 0.13, 0.09, 0.05, 0.01, -0.02, -0.06, -0.09, -0.12, -0.16, -0.20])

    topic_concentration = np.array([0.75, 0.76, 0.74, 0.72, 0.70, 0.66, 0.60, 0.56, 0.52, 0.50, 0.47, 0.45, 0.43, 0.41])
    fragmentation = 1 - topic_concentration

    return pd.DataFrame(
        {
            "day": day,
            "volume": volume,
            "comment_valence": vc,
            "reply_valence": vr,
            "sentiment_gap": vc - vr,
            "topic_concentration": topic_concentration,
            "meaning_fragmentation": fragmentation,
        }
    )


# =========================================================
# FILE HELPERS
# =========================================================
def find_comment_reply_pairs(folder: Path) -> List[Tuple[Path, Path, str]]:
    comments = sorted(folder.glob("*_comments_v2.csv"))
    pairs: List[Tuple[Path, Path, str]] = []
    for c in comments:
        base = c.name.replace("_comments_v2.csv", "")
        r = folder / f"{base}_replies_v2.csv"
        if r.exists():
            pairs.append((c, r, base))
    return pairs


def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Could not read {path}")


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if c.lower() in {"date", "created_at", "published_at", "timestamp", "time", "comment_published_at", "reply_published_at"}
    ]
    return candidates[0] if candidates else None


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if c.lower() in {"text", "comment", "comment_text", "reply_text", "content", "body"}
    ]
    return candidates[0] if candidates else None


def detect_author_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if c.lower() in {
            "author", "author_id", "author_name", "user", "user_id", "username",
            "commenter", "channel_id", "actor", "reply_author", "comment_author"
        }
    ]
    return candidates[0] if candidates else None


def detect_comment_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if c.lower() in {"comment_id", "id", "node_id", "commentid", "cid"}
    ]
    return candidates[0] if candidates else None


def detect_parent_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if c.lower() in {"parent_id", "comment_id", "reply_to", "in_reply_to", "parent_comment_id"}
    ]
    return candidates[0] if candidates else None


def detect_reply_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if c.lower() in {"reply_id", "id", "node_id", "rid"}
    ]
    return candidates[0] if candidates else None


# =========================================================
# SIMPLE SENTIMENT + DAILY AGGREGATION
# =========================================================
def simple_sentiment_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    positive_words = {"good", "great", "love", "win", "best", "strong", "happy", "support", "amazing"}
    negative_words = {"bad", "hate", "loss", "worst", "weak", "angry", "awful", "terrible", "fraud"}
    tokens = [t.strip(".,!?;:'\"()[]{}").lower() for t in text.split()]
    if not tokens:
        return 0.0
    pos = sum(t in positive_words for t in tokens)
    neg = sum(t in negative_words for t in tokens)
    return (pos - neg) / max(len(tokens), 1)


def aggregate_pair_to_daily(comment_df: pd.DataFrame, reply_df: pd.DataFrame) -> pd.DataFrame:
    c = comment_df.copy()
    r = reply_df.copy()

    c_time = detect_time_column(c)
    r_time = detect_time_column(r)
    c_text = detect_text_column(c)
    r_text = detect_text_column(r)

    if c_time is None or r_time is None:
        raise ValueError("Could not detect time columns in comments or replies.")

    c["date"] = pd.to_datetime(c[c_time], errors="coerce").dt.date
    r["date"] = pd.to_datetime(r[r_time], errors="coerce").dt.date
    c = c.dropna(subset=["date"])
    r = r.dropna(subset=["date"])

    c["sentiment"] = c[c_text].apply(simple_sentiment_score) if c_text else 0.0
    r["sentiment"] = r[r_text].apply(simple_sentiment_score) if r_text else 0.0

    c_daily = c.groupby("date").agg(comment_count=("date", "size"), comment_valence=("sentiment", "mean"))
    r_daily = r.groupby("date").agg(reply_count=("date", "size"), reply_valence=("sentiment", "mean"))

    daily = c_daily.join(r_daily, how="outer").fillna(0).reset_index()
    daily["volume"] = daily["comment_count"] + daily["reply_count"]
    daily["sentiment_gap"] = daily["comment_valence"] - daily["reply_valence"]

    # placeholder concentration / fragmentation
    daily["topic_concentration"] = np.clip(1 / (1 + daily["reply_count"] / (daily["comment_count"] + 1)), 0, 1)
    daily["meaning_fragmentation"] = 1 - daily["topic_concentration"]

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day"] = np.arange(1, len(daily) + 1)
    return daily


# =========================================================
# CLASSIFICATION
# =========================================================
def count_local_maxima(series: np.ndarray) -> int:
    if len(series) < 3:
        return 0
    count = 0
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            count += 1
    return count


def classify_wave_type(volume: pd.Series) -> str:
    y = volume.to_numpy(dtype=float)
    if len(y) < 3:
        return "Indeterminate"
    maxima = count_local_maxima(y)
    directional = y[-1] - y[0]
    if np.std(y) < 1e-6 or abs(directional) < 0.05 * max(np.max(y), 1):
        return "Drift"
    if maxima >= 3 and directional > 0:
        return "Rolling-Up"
    if maxima >= 3 and directional < 0:
        return "Rolling-Down"
    if maxima == 2 and directional > 0:
        return "Echo-Up"
    if maxima == 2 and directional < 0:
        return "Echo-Down"
    if maxima <= 1 and directional > 0:
        return "Flash-Up"
    if maxima <= 1 and directional < 0:
        return "Flash-Down"
    return "Indeterminate"


def classify_joint_valence(vc: pd.Series, vr: pd.Series) -> str:
    y1 = vc.to_numpy(dtype=float)
    y2 = vr.to_numpy(dtype=float)
    if len(y1) < 4 or len(y2) < 4:
        return "Indeterminate (Loose) Coupling"

    dv1 = y1[-1] - y1[0]
    dv2 = y2[-1] - y2[0]
    corr = np.corrcoef(y1, y2)[0, 1] if np.std(y1) > 0 and np.std(y2) > 0 else 0

    if np.argmin(y1) > 0 and y1[-1] > y1[np.argmin(y1)] and y1[-1] > y1[0] and y2[-1] >= y2[0]:
        return "Resilient Rebound"
    if dv1 > 0 and dv2 > 0 and corr > 0.2:
        return "Positive Resonance"
    if dv1 < 0 and dv2 < 0 and corr > 0.2:
        return "Negative Entrenchment"
    if (dv1 > 0 and dv2 < 0) or (dv1 < 0 and dv2 > 0) or corr < -0.2:
        return "Divergence"
    return "Indeterminate (Loose) Coupling"


def level_from_wave_and_volume(wave_type: str, volume: pd.Series) -> str:
    avg = float(volume.mean())
    peak = float(volume.max())
    if wave_type in {"Flash-Up", "Rolling-Up"} and peak > avg * 1.4:
        return "Very High"
    if avg > 15:
        return "High"
    if avg > 6:
        return "Medium"
    return "Low"


def risk_from_joint_and_fragmentation(joint_valence: str, fragmentation: pd.Series) -> str:
    frag = float(fragmentation.mean())
    if joint_valence == "Negative Entrenchment":
        return "Very High"
    if joint_valence == "Divergence":
        return "High"
    if joint_valence == "Indeterminate (Loose) Coupling":
        return "Medium"
    if joint_valence == "Resilient Rebound" and frag < 0.45:
        return "Low"
    if joint_valence == "Positive Resonance" and frag < 0.45:
        return "Low"
    return "Medium"


# =========================================================
# NETWORK BUILD
# =========================================================
def build_multilayer_network(comment_df: pd.DataFrame, reply_df: pd.DataFrame) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Build a multi-layer discourse network:
    - user -> comment node
    - replier -> reply node
    - reply node -> parent comment node
    - replier -> original commenter
    """
    G = nx.DiGraph()

    c = comment_df.copy()
    r = reply_df.copy()

    c_author = detect_author_column(c)
    r_author = detect_author_column(r)
    c_id = detect_comment_id_column(c)
    r_id = detect_reply_id_column(r)
    parent_col = detect_parent_column(r)

    if c_id is None:
        raise ValueError("Could not detect comment ID column in comments file.")
    if parent_col is None:
        raise ValueError("Could not detect parent/comment reference column in replies file.")

    if c_author is None:
        c_author = "__comment_author__"
        c[c_author] = [f"comment_user_{i}" for i in range(len(c))]

    if r_author is None:
        r_author = "__reply_author__"
        r[r_author] = [f"reply_user_{i}" for i in range(len(r))]

    c_text = detect_text_column(c)
    r_text = detect_text_column(r)

    c_lookup = c[[c_id, c_author]].copy()
    if c_text:
        c_lookup["comment_text"] = c[c_text]
    else:
        c_lookup["comment_text"] = ""

    c_lookup[c_id] = c_lookup[c_id].astype(str)
    c_lookup[c_author] = c_lookup[c_author].astype(str)

    comment_author_map = dict(zip(c_lookup[c_id], c_lookup[c_author]))
    comment_text_map = dict(zip(c_lookup[c_id], c_lookup["comment_text"]))

    for _, row in c_lookup.iterrows():
        comment_node = f"comment::{row[c_id]}"
        user_node = f"user::{row[c_author]}"

        G.add_node(user_node, node_type="user", label=str(row[c_author]))
        G.add_node(comment_node, node_type="comment", label=str(row[c_id]), text=str(row["comment_text"])[:80])
        G.add_edge(user_node, comment_node, edge_type="authored_comment")

    r = r.copy()
    r[parent_col] = r[parent_col].astype(str)
    if r_id is None:
        r_id = "__reply_id__"
        r[r_id] = [f"reply_{i}" for i in range(len(r))]

    for _, row in r.iterrows():
        parent_id = str(row[parent_col])
        if parent_id not in comment_author_map:
            continue

        reply_id = str(row[r_id])
        replier = str(row[r_author])
        original_author = str(comment_author_map[parent_id])

        reply_node = f"reply::{reply_id}"
        replier_node = f"user::{replier}"
        parent_comment_node = f"comment::{parent_id}"
        original_author_node = f"user::{original_author}"

        reply_text = str(row[r_text])[:80] if r_text and pd.notna(row[r_text]) else ""

        G.add_node(replier_node, node_type="user", label=replier)
        G.add_node(reply_node, node_type="reply", label=reply_id, text=reply_text)

        G.add_edge(replier_node, reply_node, edge_type="authored_reply")
        G.add_edge(reply_node, parent_comment_node, edge_type="replies_to_comment")

        if replier != original_author:
            G.add_edge(replier_node, original_author_node, edge_type="user_to_user_reply")

    user_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "user_to_user_reply"
    ]

    UG = nx.DiGraph()
    UG.add_edges_from(user_edges)

    if len(UG.nodes) == 0:
        metrics_df = pd.DataFrame(columns=[
            "node", "label", "degree", "betweenness", "eigenvector", "in_degree", "out_degree"
        ])
        return G, metrics_df

    degree_dict = dict(UG.degree())
    in_degree_dict = dict(UG.in_degree())
    out_degree_dict = dict(UG.out_degree())
    betweenness_dict = nx.betweenness_centrality(UG) if len(UG) > 1 else {n: 0.0 for n in UG.nodes}

    try:
        eigenvector_dict = nx.eigenvector_centrality_numpy(UG.to_undirected()) if len(UG) > 1 else {n: 1.0 for n in UG.nodes}
    except Exception:
        eigenvector_dict = {n: 0.0 for n in UG.nodes}

    metrics_df = pd.DataFrame({
        "node": list(UG.nodes),
        "label": [n.replace("user::", "") for n in UG.nodes],
        "degree": [degree_dict.get(n, 0) for n in UG.nodes],
        "betweenness": [betweenness_dict.get(n, 0.0) for n in UG.nodes],
        "eigenvector": [eigenvector_dict.get(n, 0.0) for n in UG.nodes],
        "in_degree": [in_degree_dict.get(n, 0) for n in UG.nodes],
        "out_degree": [out_degree_dict.get(n, 0) for n in UG.nodes],
    }).sort_values(["betweenness", "eigenvector", "degree"], ascending=False)

    return G, metrics_df


def make_user_interaction_subgraph(multilayer_graph: nx.DiGraph) -> nx.Graph:
    edges = [
        (u, v) for u, v, d in multilayer_graph.edges(data=True)
        if d.get("edge_type") == "user_to_user_reply"
    ]
    H = nx.Graph()
    H.add_edges_from(edges)
    return H


def draw_network_plotly(multilayer_graph: nx.DiGraph, top_n: int = 28) -> go.Figure:
    H = make_user_interaction_subgraph(multilayer_graph)

    if len(H.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[dict(
                text="No user-to-user reply network found.",
                showarrow=False,
                x=0.5, y=0.5, xref="paper", yref="paper"
            )]
        )
        return fig

    degree_rank = sorted(H.degree, key=lambda x: x[1], reverse=True)
    keep_nodes = [n for n, _ in degree_rank[:top_n]]
    H = H.subgraph(keep_nodes).copy()

    pos = nx.spring_layout(H, seed=42, k=0.85)

    edge_x = []
    edge_y = []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#c7c7c7"),
        hoverinfo="none"
    )

    degree_dict = dict(H.degree())
    betweenness_dict = nx.betweenness_centrality(H) if len(H) > 1 else {n: 0.0 for n in H.nodes}
    try:
        eigenvector_dict = nx.eigenvector_centrality_numpy(H) if len(H) > 1 else {n: 1.0 for n in H.nodes}
    except Exception:
        eigenvector_dict = {n: 0.0 for n in H.nodes}

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []

    for node in H.nodes():
        x, y = pos[node]
        label = node.replace("user::", "")
        deg = degree_dict.get(node, 0)
        btw = betweenness_dict.get(node, 0.0)
        eig = eigenvector_dict.get(node, 0.0)

        node_x.append(x)
        node_y.append(y)
        node_size.append(9 + deg * 2.0)
        node_color.append(btw)
        node_text.append(
            f"{label}<br>"
            f"Degree: {deg}<br>"
            f"Bridging: {btw:.3f}<br>"
            f"Impact: {eig:.3f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Bridging"),
            line=dict(width=0.8, color="white")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def summarize_network_metrics(metrics_df: pd.DataFrame) -> dict:
    if metrics_df.empty:
        return {
            "n_users": 0,
            "degree": 0.0,
            "bridging": 0.0,
            "impact": 0.0,
        }

    return {
        "n_users": int(metrics_df["node"].nunique()),
        "degree": float(metrics_df["degree"].mean()),
        "bridging": float(metrics_df["betweenness"].mean()),
        "impact": float(metrics_df["eigenvector"].mean()),
    }


# =========================================================
# UI HELPERS
# =========================================================
def show_metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="small-card">
            <div style="font-size:0.70rem;color:#666;">{label}</div>
            <div style="font-size:0.86rem;font-weight:800;margin-top:4px;line-height:1.1;">{value}</div>
            <div style="font-size:0.64rem;color:#777;margin-top:5px;line-height:1.05;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_lines(df: pd.DataFrame, x_col: str, cols: List[str], title: str, height: int = 145):
    chart_df = df[[x_col] + cols].copy().set_index(x_col)
    st.line_chart(chart_df, height=height)
    st.caption(title)


def build_wave_shape_series(wave_type: str) -> pd.DataFrame:
    shapes = {
        "Rolling-Up": [2, 4, 6, 8, 7, 6, 9],
        "Rolling-Down": [9, 8, 6, 6, 4, 2, 1],
        "Echo-Up": [3, 5, 6, 8, 5, 6],
        "Echo-Down": [8, 7, 5, 8, 5, 3],
        "Flash-Up": [3, 5, 10, 10, 10, 5, 7],
        "Flash-Down": [10, 10, 6, 3, 1, 1, 1],
        "Drift": [5, 5, 5, 5, 5, 5],
        "Indeterminate": [5, 5, 5, 5, 5, 5],
    }
    y = shapes.get(wave_type, shapes["Indeterminate"])
    return pd.DataFrame({"t": range(1, len(y) + 1), "wave": y})


def build_joint_valence_shape(joint_valence: str) -> pd.DataFrame:
    shapes = {
        "Positive Resonance": ([3, 5, 6, 7, 9], [4, 6, 7, 8, 9]),
        "Negative Entrenchment": ([9, 7, 6, 4, 2], [8, 6, 5, 3, 1]),
        "Resilient Rebound": ([8, 5, 3, 4, 6, 8], [5, 4, 5, 6, 7, 7]),
        "Divergence": ([3, 4, 5, 6, 7], [7, 6, 5, 4, 2]),
        "Indeterminate (Loose) Coupling": ([5, 6, 5, 6, 5], [4, 3, 4, 3, 4]),
    }
    vc, vr = shapes.get(joint_valence, shapes["Indeterminate (Loose) Coupling"])
    n = max(len(vc), len(vr))
    vc_full = vc + [vc[-1]] * (n - len(vc))
    vr_full = vr + [vr[-1]] * (n - len(vr))
    return pd.DataFrame({"t": range(1, n + 1), "VC": vc_full, "VR": vr_full})


def show_process_images():
    existing = {k: v for k, v in IMAGE_FILES.items() if Path(v).exists()}
    if not existing:
        st.info("Place your exported process images in the same folder as this app to display them here.")
        return
    tabs = st.tabs([k.replace("_", " ").title() for k in existing.keys()])
    for tab, (_, file) in zip(tabs, existing.items()):
        with tab:
            st.image(file, use_container_width=True)


# =========================================================
# APP
# =========================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Choose mode", ["Demo", "EPL folder"], index=0)

    epl_folder = None
    if mode == "EPL folder":
        st.markdown("Use a local folder path that contains paired `*_comments_v2.csv` and `*_replies_v2.csv` files.")
        epl_folder = st.text_input(
            "EPL threaded folder",
            value=str(Path(__file__).resolve().parent / "data" / "threaded" / "epl"),
        )

main_tabs = st.tabs(["Explorer", "Process images", "Operationalization table"])

with main_tabs[2]:
    st.subheader("Variable operationalization table")
    st.dataframe(build_operationalization_table(), use_container_width=True, height=520)

with main_tabs[1]:
    st.subheader("Process image library")
    show_process_images()

with main_tabs[0]:
    network_graph = None
    network_metrics_df = pd.DataFrame()
    network_summary = {"n_users": 0, "degree": 0.0, "bridging": 0.0, "impact": 0.0}

    if mode == "Demo":
        df = make_demo_time_series()
        dataset_name = "Demo dataset"
    else:
        folder = Path(epl_folder) if epl_folder else None
        if folder is None or not folder.exists():
            st.warning("Enter a valid EPL folder path to load local files.")
            st.stop()

        pairs = find_comment_reply_pairs(folder)
        if not pairs:
            st.error("No matching comment/reply file pairs found in that folder.")
            st.stop()

        pair_labels = [base for _, _, base in pairs]
        chosen = st.sidebar.selectbox("Choose EPL item", pair_labels)
        comment_path, reply_path, dataset_name = next((c, r, b) for c, r, b in pairs if b == chosen)

        comments_df = safe_read_csv(comment_path)
        replies_df = safe_read_csv(reply_path)
        df = aggregate_pair_to_daily(comments_df, replies_df)

        try:
            network_graph, network_metrics_df = build_multilayer_network(comments_df, replies_df)
            network_summary = summarize_network_metrics(network_metrics_df)
        except Exception as e:
            st.warning(f"Network could not be built for this item: {e}")

    wave_type = classify_wave_type(df["volume"])
    joint_valence = classify_joint_valence(df["comment_valence"], df["reply_valence"])
    level = level_from_wave_and_volume(wave_type, df["volume"])
    risk = risk_from_joint_and_fragmentation(joint_valence, df["meaning_fragmentation"])

    initial_avg = float(df["volume"].head(min(3, len(df))).mean()) if len(df) else 0.0
    initial_label = (
        "Very High volume" if initial_avg > 50 else
        "High volume" if initial_avg > 15 else
        "Medium volume" if initial_avg > 6 else
        "Low volume"
    )

    st.subheader(f"Dataset: {dataset_name}")
    st.caption("One-video workflow: starting condition, volume path, meaning path, network effect, and final readout.")

    step_colors = [
        ("#1f4ed8", "#eff6ff"),
        ("#0f766e", "#f0fdfa"),
        ("#6d28d9", "#f5f3ff"),
        ("#ea580c", "#fff7ed"),
        ("#2563eb", "#eff6ff"),
        ("#b45309", "#fffbeb"),
        ("#065f46", "#ecfdf5"),
    ]

    flow_cols = st.columns(7)
    flow_items = [
        ("1. Initial Volume", "Baseline", initial_label),
        ("2. Inflow Structure", "Hidden drivers", "Demo scaffold" if mode == "Demo" else f"Users {network_summary['n_users']}"),
        ("3. Volume Change", "Activity trajectory", wave_type),
        ("4. Valence Change", "Meaning shift", f"Gap {df['sentiment_gap'].mean():.2f}"),
        ("5. Wave Type", "Surface trajectory", wave_type),
        ("6. Joint Valence", "Meaning trajectory", joint_valence),
        ("7. Final Readout", "Manager dashboard", f"Level {level} / Risk {risk}"),
    ]

    for idx, (col, (title, subtitle, value)) in enumerate(zip(flow_cols, flow_items)):
        accent, bg = step_colors[idx]
        with col:
            st.markdown(
                f"""
                <div class="flow-card" style="border-color:{accent};background:{bg};">
                    <div class="flow-num" style="background:{accent};">{idx+1}</div>
                    <div class="flow-title" style="color:{accent};">{title}</div>
                    <div class="flow-sub">{subtitle}</div>
                    <div class="flow-val">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------
    # PATH + VOLUME ONLY
    # -----------------------------------------------------
    st.markdown("### Selected video path")
    path_left, path_right = st.columns([1.55, 0.95])

    with path_left:
        st.markdown(
            f"""
            <div style="padding:8px 10px;border:1px solid #d9d9d9;border-radius:12px;background:#ffffff;">
                <div style="font-size:0.82rem;font-weight:800;">Path for this video</div>
                <div style="font-size:0.72rem;color:#555;margin-top:4px;line-height:1.25;">
                    Start: <b>{initial_label}</b> → Volume path: <b>{wave_type}</b> → Meaning path: <b>{joint_valence}</b> → Final: <b>Level {level} / Risk {risk}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Volume for this video")
        plot_lines(
            df,
            "day",
            ["volume"],
            "Selected video: volume only.",
            height=150,
        )

    with path_right:
        c1, c2 = st.columns(2)
        with c1:
            show_metric_card("Initial", initial_label, "Starting attention")
        with c2:
            show_metric_card("Final", f"Level {level} / Risk {risk}", "Manager summary")

        c3, c4 = st.columns(2)
        with c3:
            show_metric_card("Wave", wave_type, "Volume path")
        with c4:
            show_metric_card("Joint valence", joint_valence, "Meaning path")

        st.markdown("#### Reference shapes")
        s1, s2 = st.columns(2)
        with s1:
            st.caption("Wave type")
            st.line_chart(build_wave_shape_series(wave_type).set_index("t"), height=95)
        with s2:
            st.caption("Joint valence")
            st.line_chart(build_joint_valence_shape(joint_valence).set_index("t"), height=95)

    # -----------------------------------------------------
    # NETWORK
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### Network effect for this video")
    net_left, net_right = st.columns([0.78, 1.22])

    with net_left:
        if mode == "Demo":
            st.markdown(
                """
                <div style="padding:10px;border:1px solid #d9d9d9;border-radius:12px;background:#faf7ff;min-height:220px;">
                    <div style="font-size:0.82rem;font-weight:800;color:#5b21b6;">Network guide</div>
                    <div style="margin-top:8px;font-size:0.70rem;"><b>Degree</b><br>Local interaction intensity</div>
                    <div style="margin-top:7px;font-size:0.70rem;"><b>Bridging</b><br>Cross-group brokerage</div>
                    <div style="margin-top:7px;font-size:0.70rem;"><b>Impact</b><br>Connection to influential users</div>
                    <div style="margin-top:7px;font-size:0.70rem;"><b>Amplification</b><br>Excess visibility beyond structure</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            r1, r2 = st.columns(2)
            with r1:
                show_metric_card("Users", str(network_summary["n_users"]), "Active users")
            with r2:
                show_metric_card("Degree", f"{network_summary['degree']:.2f}", "Interaction intensity")

            r3, r4 = st.columns(2)
            with r3:
                show_metric_card("Bridging", f"{network_summary['bridging']:.3f}", "Cross-group brokerage")
            with r4:
                show_metric_card("Impact", f"{network_summary['impact']:.3f}", "Influence score")

            if not network_metrics_df.empty:
                st.markdown("##### Top users")
                st.dataframe(
                    network_metrics_df[["label", "degree", "betweenness", "eigenvector"]].head(6),
                    use_container_width=True,
                    height=155,
                )
            else:
                st.info("No user-to-user reply network found.")

    with net_right:
        if mode == "Demo":
            st.info("In EPL mode this panel will draw the actual user-reply network.")
        else:
            fig_net = draw_network_plotly(network_graph, top_n=28)
            fig_net.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": False})

    # -----------------------------------------------------
    # VALENCE / MEANING
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### Valence and meaning indicators")
    val_left, val_right = st.columns(2)

    with val_left:
        plot_lines(
            df,
            "day",
            ["comment_valence", "reply_valence", "sentiment_gap"],
            "Comment valence, reply valence, and sentiment gap.",
            height=145,
        )

    with val_right:
        plot_lines(
            df,
            "day",
            ["topic_concentration", "meaning_fragmentation"],
            "Topic concentration and meaning fragmentation.",
            height=145,
        )

    # -----------------------------------------------------
    # DAILY DATA
    # -----------------------------------------------------
    with st.expander("Daily data", expanded=False):
        st.dataframe(df, use_container_width=True, height=170)

    # -----------------------------------------------------
    # NOTE
    # -----------------------------------------------------
    st.markdown("---")
    a, b = st.columns(2)
    with a:
        st.info("Demo mode shows the concept immediately: volume → wave type, VC/VR → joint valence, then Level + Risk.")
    with b:
        st.info("EPL mode applies the same workflow to your paired local files and adds the real user-reply network.")
