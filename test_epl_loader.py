import os
import pandas as pd


def load_epl_threaded_data(folder_path):
    print("Looking in:", folder_path)

    all_data = []

    all_files = os.listdir(folder_path)
    print("All files in folder:", all_files[:10], "... total:", len(all_files))

    comment_files = [
        os.path.join(folder_path, f)
        for f in all_files
        if "comments_v2" in f.lower()
        and (
            f.lower().endswith(".csv")
            or f.lower().endswith(".xlsx")
            or f.lower().endswith(".xls")
        )
    ]

    print("Comment files found:", len(comment_files))

    for comment_file in comment_files[:3]:   # first 3 only for testing
        print("\n---")
        print("Loading comment file:", comment_file)

        try:
            if comment_file.lower().endswith(".csv"):
                try:
                    df_comments = pd.read_csv(comment_file, encoding="utf-8", on_bad_lines="skip")
                except Exception:
                    df_comments = pd.read_csv(comment_file, encoding="latin1", on_bad_lines="skip")
            else:
                df_comments = pd.read_excel(comment_file)

            print("Comment shape:", df_comments.shape)
            print("Comment columns:", df_comments.columns.tolist())

            if df_comments.empty:
                print("Skipping empty comment file")
                continue

        except Exception as e:
            print("Skipping bad comment file")
            print("Reason:", str(e))
            continue

        base_name = (
            comment_file
            .replace("_comments_v2.xlsx", "")
            .replace("_comments_v2.xls", "")
            .replace("_comments_v2.csv", "")
        )

        reply_file_xlsx = base_name + "_replies_v2.xlsx"
        reply_file_xls = base_name + "_replies_v2.xls"
        reply_file_csv = base_name + "_replies_v2.csv"

        if os.path.exists(reply_file_xlsx):
            reply_file = reply_file_xlsx
        elif os.path.exists(reply_file_xls):
            reply_file = reply_file_xls
        elif os.path.exists(reply_file_csv):
            reply_file = reply_file_csv
        else:
            reply_file = None

        print("Matching reply file:", reply_file)

        # Rename comments
        df_comments = df_comments.rename(columns={
            df_comments.columns[1]: "node_id",
            df_comments.columns[2]: "text",
            df_comments.columns[3]: "user",
            df_comments.columns[4]: "timestamp"
        })
        df_comments["parent_id"] = None
        df_comments["type"] = "comment"

        if reply_file is not None:
            try:
                if reply_file.lower().endswith(".csv"):
                    try:
                        df_replies = pd.read_csv(reply_file, encoding="utf-8", on_bad_lines="skip")
                    except Exception:
                        df_replies = pd.read_csv(reply_file, encoding="latin1", on_bad_lines="skip")
                else:
                    df_replies = pd.read_excel(reply_file)

                print("Reply shape:", df_replies.shape)
                print("Reply columns:", df_replies.columns.tolist())

                if not df_replies.empty:
                    df_replies = df_replies.rename(columns={
                        df_replies.columns[1]: "parent_id",
                        df_replies.columns[2]: "node_id",
                        df_replies.columns[3]: "text",
                        df_replies.columns[4]: "user",
                        df_replies.columns[5]: "timestamp"
                    })
                    df_replies["type"] = "reply"
                    combined = pd.concat([df_comments, df_replies], ignore_index=True)
                else:
                    combined = df_comments

            except Exception as e:
                print("Skipping bad reply file")
                print("Reason:", str(e))
                combined = df_comments
        else:
            combined = df_comments

        print("Combined shape:", combined.shape)
        print(combined.head())

        all_data.append(combined)

    if not all_data:
        print("\nNo usable EPL files found.")
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    print("\nFinal combined shape:", full_df.shape)
    return full_df


def convert_epl_to_dhs(df):
    out = df.copy()

    out["post_id"] = out["node_id"]
    out["reply_to_post_id"] = out["parent_id"]

    # each video becomes its own event
    out["event_name"] = out["video_id"].astype(str)

    out["user_id"] = out["user"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # basic depth for now
    out["reply_depth"] = out["type"].apply(lambda x: 1 if x == "reply" else 0)

    # placeholders for now
    out["comment_sentiment"] = 0.0
    out["reply_sentiment"] = 0.0
    out["topic_similarity"] = 0.5
    out["topic_entropy"] = 0.5

    out["platform"] = "YouTube"
    out["source_type"] = "public"
    out["is_elite_source"] = 0
    out["cluster_id"] = "Cluster 1"

    return out[[
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
        "platform",
        "source_type",
        "is_elite_source",
        "cluster_id",
    ]]