import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DHS / DSH Explorer", layout="wide")

# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "Discourse Dynamics Explorer"
APP_SUBTITLE = "Demo + EPL workflow for Level, Risk, Wave Type, Joint Valence, and Network Snapshot"

# Put your generated process images in the same folder as this app,
# or update these paths to match your local machine.
IMAGE_FILES = {
    "overall_flow": "overall_process_flow.png",
    "paths_1_4": "paths_1_4.png",
    "network_snapshot": "network_snapshot.png",
    "volume_vs_valence": "volume_vs_valence.png",
    "wave_vs_joint": "wave_vs_joint.png",
}

# =========================================================
# VARIABLE OPERATIONALIZATION TABLE
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
            "Operationalization": "Time series of comments + replies by day; optionally likes/retweets if available.",
            "Data source": "comments + replies files",
            "Unit": "daily count",
            "Interpretation": "Used to classify wave types.",
        },
        {
            "Layer": "Surface / Level",
            "Variable": "Wave Type",
            "Definition": "Temporal geometry of attention over time.",
            "Operationalization": "Classify by number of local maxima and directional change (last - first): Rolling-Up, Rolling-Down, Echo-Up, Echo-Down, Flash-Up, Flash-Down, Drift.",
            "Data source": "derived from daily volume",
            "Unit": "category",
            "Interpretation": "Represents visible momentum / Level.",
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
            "Operationalization": "VC - VR by period or rolling window.",
            "Data source": "derived from VC and VR",
            "Unit": "difference score",
            "Interpretation": "Larger gap indicates stronger misalignment or opposition.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Topic Concentration",
            "Definition": "How focused discussion is on a small number of topics.",
            "Operationalization": "1 - topic entropy, or concentration index based on topic shares.",
            "Data source": "comments + replies text",
            "Unit": "index",
            "Interpretation": "Higher values mean discussion stays focused.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Meaning Fragmentation",
            "Definition": "Whether discussion splits into multiple disconnected interpretations.",
            "Operationalization": "Semantic cluster count / semantic dispersion / topic entropy.",
            "Data source": "comments + replies text",
            "Unit": "index",
            "Interpretation": "Higher values mean scattered meanings.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Polarization",
            "Definition": "Degree of division into opposing camps on the same issue.",
            "Operationalization": "Distance or bimodality across sentiment / stance groups.",
            "Data source": "comments + replies text",
            "Unit": "index",
            "Interpretation": "Higher values mean stronger opposing camps.",
        },
        {
            "Layer": "Meaning / Risk",
            "Variable": "Joint Valence",
            "Definition": "Co-evolutionary pattern between comment valence (VC) and reply valence (VR).",
            "Operationalization": "Classify by directional relation between VC and VR: Positive Resonance, Negative Entrenchment, Resilient Rebound, Divergence, Indeterminate (Loose) Coupling.",
            "Data source": "derived from VC and VR time series",
            "Unit": "category",
            "Interpretation": "Represents hidden instability / Risk.",
        },
        {
            "Layer": "Network / Inflow",
            "Variable": "Degree",
            "Definition": "Local interaction intensity around a node or discussion.",
            "Operationalization": "Average degree, degree centralization, or top-decile degree across participants.",
            "Data source": "thread / reply network",
            "Unit": "network metric",
            "Interpretation": "Higher values mean stronger within-cluster visibility.",
        },
        {
            "Layer": "Network / Inflow",
            "Variable": "Bridging (Betweenness)",
            "Definition": "Cross-group brokerage connecting otherwise separate communities.",
            "Operationalization": "Betweenness centrality or share of cross-cluster edges.",
            "Data source": "thread / reply network",
            "Unit": "network metric",
            "Interpretation": "Higher values mean new attention enters across groups.",
        },
        {
            "Layer": "Network / Inflow",
            "Variable": "Impact (Eigenvector)",
            "Definition": "Influence through connection to already influential nodes.",
            "Operationalization": "Eigenvector centrality or concentration among top nodes.",
            "Data source": "thread / reply network",
            "Unit": "network metric",
            "Interpretation": "Higher values mean stronger hub-driven guidance of conversation.",
        },
        {
            "Layer": "Network / Inflow",
            "Variable": "Amplification (Residual / AI-mediated)",
            "Definition": "Extra visibility not explained by degree, bridging, or impact.",
            "Operationalization": "Residual reach / excess exposure after predicting attention from network structure.",
            "Data source": "derived model output",
            "Unit": "residual index",
            "Interpretation": "Higher values imply algorithmic or external boosting beyond network position.",
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
            "Operationalization": "Derived from joint valence, fragmentation, polarization, and misalignment.",
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

    df = pd.DataFrame({
        "day": day,
        "volume": volume,
        "comment_valence": vc,
        "reply_valence": vr,
        "sentiment_gap": vc - vr,
        "topic_concentration": topic_concentration,
        "meaning_fragmentation": fragmentation,
    })
    return df


# =========================================================
# EPL DATA LOADING
# =========================================================

def find_comment_reply_pairs(folder: Path) -> List[Tuple[Path, Path, str]]:
    comments = sorted(folder.glob("*_comments_v2.csv"))
    pairs = []
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
    candidates = [c for c in df.columns if c.lower() in {
        "date", "created_at", "published_at", "timestamp", "time", "comment_published_at"
    }]
    if candidates:
        return candidates[0]
    return None


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {
        "text", "comment", "comment_text", "reply_text", "content", "body"
    }]
    return candidates[0] if candidates else None


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

    # Placeholder topic-focused measures for first build
    daily["topic_concentration"] = np.clip(1 / (1 + daily["reply_count"] / (daily["comment_count"] + 1)), 0, 1)
    daily["meaning_fragmentation"] = 1 - daily["topic_concentration"]
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day"] = np.arange(1, len(daily) + 1)
    return daily


# =========================================================
# CLASSIFIERS
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
    return "Drift"


def classify_joint_valence(vc: pd.Series, vr: pd.Series) -> str:
    y1 = vc.to_numpy(dtype=float)
    y2 = vr.to_numpy(dtype=float)
    if len(y1) < 4 or len(y2) < 4:
        return "Indeterminate (Loose) Coupling"

    dv1 = y1[-1] - y1[0]
    dv2 = y2[-1] - y2[0]
    corr = np.corrcoef(y1, y2)[0, 1] if np.std(y1) > 0 and np.std(y2) > 0 else 0

    # Resilient rebound: comments dip then recover, replies stabilize or recover
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
# UI HELPERS
# =========================================================

def show_metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding:14px;border:1px solid #ddd;border-radius:14px;background:#fafafa;">
            <div style="font-size:0.9rem;color:#666;">{label}</div>
            <div style="font-size:1.35rem;font-weight:700;margin-top:6px;">{value}</div>
            <div style="font-size:0.8rem;color:#777;margin-top:6px;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_lines(df: pd.DataFrame, x_col: str, cols: List[str], title: str):
    chart_df = df[[x_col] + cols].copy().set_index(x_col)
    st.line_chart(chart_df, height=280)
    st.caption(title)


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
    selected_pair = None
    if mode == "EPL folder":
        st.markdown("Use a local folder path on your machine that contains paired `*_comments_v2.csv` and `*_replies_v2.csv` files.")
        epl_folder = st.text_input(
            "EPL threaded folder",
            value=str(Path(__file__).resolve().parent / "data" / "threaded" / "epl")
        )

# Top tabs
main_tabs = st.tabs([
    "Explorer",
    "Process images",
    "Operationalization table",
])

with main_tabs[2]:
    st.subheader("Variable operationalization table")
    st.dataframe(build_operationalization_table(), use_container_width=True, height=650)

with main_tabs[1]:
    st.subheader("Process image library")
    show_process_images()

with main_tabs[0]:
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

    wave_type = classify_wave_type(df["volume"])
    joint_valence = classify_joint_valence(df["comment_valence"], df["reply_valence"])
    level = level_from_wave_and_volume(wave_type, df["volume"])
    risk = risk_from_joint_and_fragmentation(joint_valence, df["meaning_fragmentation"])

    st.subheader(f"Dataset: {dataset_name}")
    st.caption("This panel follows one video from start to finish: starting condition, trajectory, meaning shift, network view, and indicator guide.")

    # ---------- Process flow for the selected video ----------
    step_colors = [
        ("#1f4ed8", "#eff6ff"),
        ("#0f766e", "#f0fdfa"),
        ("#6d28d9", "#f5f3ff"),
        ("#ea580c", "#fff7ed"),
        ("#2563eb", "#eff6ff"),
        ("#b45309", "#fffbeb"),
        ("#065f46", "#ecfdf5"),
    ]

    initial_avg = float(df["volume"].head(min(3, len(df))).mean()) if len(df) else 0.0
    initial_label = "Very High volume" if initial_avg > 50 else "High volume" if initial_avg > 15 else "Medium volume" if initial_avg > 6 else "Low volume"

    flow_cols = st.columns(7)
    flow_items = [
        ("1. Initial Volume", "Baseline", initial_label),
        ("2. Inflow Structure", "Hidden drivers", "Demo scaffold" if mode == "Demo" else "EPL local files"),
        ("3. Volume Change", "Activity trajectory", wave_type),
        ("4. Valence Change", "Meaning shift", f"VC / VR gap {df['sentiment_gap'].mean():.2f}"),
        ("5. Wave Type", "Surface trajectory", wave_type),
        ("6. Joint Valence", "Meaning trajectory", joint_valence),
        ("7. Final Readout", "Manager dashboard", f"Level {level} / Risk {risk}"),
    ]

    for idx, (col, (title, subtitle, value)) in enumerate(zip(flow_cols, flow_items)):
        accent, bg = step_colors[idx]
        with col:
            st.markdown(
                f"""
                <div style="padding:12px;border:1.5px solid {accent};border-radius:16px;background:{bg};min-height:188px;">
                    <div style="width:30px;height:30px;border-radius:999px;background:{accent};color:white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;">{idx+1}</div>
                    <div style="font-size:0.96rem;font-weight:800;line-height:1.2;color:{accent};margin-top:10px;">{title}</div>
                    <div style="font-size:0.80rem;color:#666;margin-top:4px;">{subtitle}</div>
                    <div style="font-size:1.02rem;font-weight:800;margin-top:22px;color:#222;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Selected video path")
    path_left, path_right = st.columns([1.15, 0.85])
    with path_left:
        st.markdown(
            f"""
            <div style="padding:14px;border:1px solid #d9d9d9;border-radius:16px;background:#ffffff;">
                <div style="font-size:1rem;font-weight:800;">Path for this video</div>
                <div style="font-size:0.92rem;color:#555;margin-top:8px;">
                    Start: <b>{initial_label}</b> → Volume path: <b>{wave_type}</b> → Meaning path: <b>{joint_valence}</b> → Final: <b>Level {level} / Risk {risk}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Volume + valence for this video")
        plot_lines(df, "day", ["volume", "comment_valence", "reply_valence"], "Selected video: volume, comment valence, and reply valence.")
    with path_right:
        show_metric_card("Initial condition", initial_label, "Starting attention level for this selected video")
        show_metric_card("Wave type", wave_type, "Temporal shape of attention")
        show_metric_card("Joint valence", joint_valence, "Comment-reply affective coupling")
        show_metric_card("Final readout", f"Level {level} / Risk {risk}", "Manager-facing summary")

    st.markdown("---")
    st.markdown("### How the network looks for this video")
    net_left, net_right = st.columns([1, 1])
    with net_left:
        st.markdown(
            """
            <div style="padding:16px;border:1px solid #d9d9d9;border-radius:16px;background:#faf7ff;min-height:320px;">
                <div style="font-size:1rem;font-weight:800;color:#5b21b6;">Network snapshot (inflow structure)</div>
                <div style="margin-top:14px;font-size:0.95rem;"><b>Cluster (Community)</b><br>Tightly connected groups with dense internal interactions.</div>
                <div style="margin-top:14px;font-size:0.95rem;"><b>Bridge (Betweenness)</b><br>Connections between groups that move information across communities.</div>
                <div style="margin-top:14px;font-size:0.95rem;"><b>Influential Node (Eigenvector)</b><br>Well-connected to other influential nodes; high overall reach.</div>
                <div style="margin-top:14px;font-size:0.95rem;"><b>Amplification (Residual / AI-mediated)</b><br>Extra reach not explained by network position alone.</div>
                <div style="margin-top:18px;font-size:0.9rem;color:#6b7280;">This network panel should be read as the structural input behind the selected video path.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with net_right:
        if Path(IMAGE_FILES.get("network_snapshot", "")).exists():
            st.image(IMAGE_FILES["network_snapshot"], use_container_width=True)
        else:
            st.markdown(
                """
                <div style="padding:16px;border:1px dashed #c4b5fd;border-radius:16px;background:#fcfcff;min-height:320px;display:flex;align-items:center;justify-content:center;color:#6b7280;">
                    Add <code>network_snapshot.png</code> next to this app to show the network guide here.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Key indicators: volume versus valence")
    kv_left, kv_right = st.columns(2)
    with kv_left:
        if Path(IMAGE_FILES.get("volume_vs_valence", "")).exists():
            st.image(IMAGE_FILES["volume_vs_valence"], use_container_width=True)
        else:
            plot_lines(df, "day", ["volume"], "Add volume_vs_valence.png to replace this placeholder.")
    with kv_right:
        plot_lines(df, "day", ["comment_valence", "reply_valence", "sentiment_gap", "topic_concentration", "meaning_fragmentation"], "VC, VR, gap, concentration, and fragmentation for the selected video.")

    st.markdown("### Daily data")
    st.dataframe(df, use_container_width=True, height=240)

    st.markdown("---")
    st.markdown("### Demo + EPL interpretation workflow")
    a, b = st.columns(2)
    with a:
        st.info(
            "**Demo mode** lets you see the concept working immediately: volume → wave type, VC/VR → joint valence, then Level + Risk."
        )
    with b:
        st.info(
            "**EPL mode** applies the same workflow to your local paired files. Start with one video pair, then compare items across the folder."
        )

    st.markdown("### Recommended next build")
    st.markdown("""
1. Add real sentiment model instead of the placeholder lexicon.  
2. Build topic concentration / fragmentation from embeddings or topic model.  
3. Build reply-network metrics for degree, betweenness, eigenvector, and amplification residual.  
4. Add side-by-side comparison for Demo vs EPL item.
""")
