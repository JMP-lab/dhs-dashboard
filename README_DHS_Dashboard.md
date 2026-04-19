# DHS Streamlit Dashboard

## Files
- `dhs_dashboard_app.py` — Streamlit app
- `README_DHS_Dashboard.md` — quick start guide

## Run locally
```bash
pip install streamlit pandas numpy plotly networkx
streamlit run dhs_dashboard_app.py
```

## What the app includes
- Demo mode with synthetic data for an oil-shock discourse event
- CSV upload mode for your own data
- DHS score with system-state gauge
- Perspective Divergence panel
- Interaction Structure network graph
- Temporal Instability panel
- Early warning signals and intervention window

## Required CSV columns
```text
timestamp,event_name,user_id,post_id,reply_to_post_id,reply_depth,comment_sentiment,reply_sentiment,topic_similarity,topic_entropy
```

## Optional CSV columns
```text
platform,source_type,is_elite_source,cluster_id
```

## Notes
- `comment_sentiment` and `reply_sentiment` should typically be scaled between -1 and 1.
- `topic_similarity` should usually be between 0 and 1.
- `topic_entropy` should usually be scaled between 0 and 1 for cleaner interpretation.
- `reply_to_post_id` should be blank for top-level comments.

## Suggested next upgrades
- Add a forecasting model for DHS(t+1)
- Connect to X/TikTok pipelines
- Add event comparison mode
- Add exportable PDF or slide snapshot
