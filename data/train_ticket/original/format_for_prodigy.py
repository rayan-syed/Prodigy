import os
import pandas as pd

# === CONFIG ===
DATA_DIR = "."  
TRAIN_FILE = "train.csv"
TEST_ABN_FILE = "test_abnormal.csv"
TEST_NORM_FILE = "test_normal.csv"

# Output filenames 
OUT_DIR = os.path.join(DATA_DIR, "../for_prodigy")
OUT_TRAIN_HDF = os.path.join(OUT_DIR, "prod_train_data.hdf")
OUT_TEST_HDF = os.path.join(OUT_DIR, "prod_test_data.hdf")
OUT_TEST_LABELS = os.path.join(OUT_DIR, "prod_test_label.csv")

# ==============

def _read_no_header_csv(path):
    return pd.read_csv(path, header=None, skip_blank_lines=True)

def _split_features_and_label(df, has_label=False):
    trace_id = df.iloc[:, 0].astype(str)
    if has_label:
        # features: coerce to numeric, then fill blanks/NaNs with 0
        X = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # label: coerce to int, blanks -> 0 (healthy)
        y = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0).astype(int)
    else:
        X = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = None
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    return trace_id, X, y

def _make_ids(trace_id_series, prefix):
    """
    Build IDs that Prodigy won't choke on.
    - job_id: always "42"
    - component_id: prefix + sequential number (no underscores)
    - timestamp: "42"
    """
    n = len(trace_id_series)
    # make a sequential index 0..n-1
    seq = pd.RangeIndex(n).astype(str)
    # IMPORTANT: Series (not Index), same original index, NO underscores in component_id value
    cid = pd.Series([f"{prefix}{i}" for i in seq], index=trace_id_series.index, name="component_id")
    jid = pd.Series(["42"] * n, index=trace_id_series.index, name="job_id")
    ts  = pd.Series(["42"] * n, index=trace_id_series.index, name="timestamp")
    return jid, cid, ts


def build_train(data_dir):
    path = os.path.join(data_dir, TRAIN_FILE)
    df_raw = _read_no_header_csv(path)
    trace_id, X, _ = _split_features_and_label(df_raw, has_label=False)
    jid, cid, ts = _make_ids(trace_id, prefix="tr")
    df_train = pd.concat([jid, cid, ts, X], axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    df_train.to_hdf(OUT_TRAIN_HDF, key="prod_train_data", mode="w", format="table")
    print(f"Wrote {OUT_TRAIN_HDF}  shape={df_train.shape}")

    # NEW: training labels (all healthy)
    train_labels_path = os.path.join(OUT_DIR, "prod_train_label.csv")
    train_labels = pd.DataFrame({
        "job_id": df_train["job_id"].astype(str),
        "component_id": df_train["component_id"].astype(str),
        "binary_anom": 0,
        "app_name": "eclipse",
        "anom_name": "none",
    })
    train_labels.to_csv(train_labels_path, index=False)
    print(f"Wrote {train_labels_path}  shape={train_labels.shape}")


def build_test_and_labels(data_dir):
    # abnormal
    abn_path = os.path.join(data_dir, TEST_ABN_FILE)
    df_abn_raw = _read_no_header_csv(abn_path)
    abn_trace, X_abn, y_abn = _split_features_and_label(df_abn_raw, has_label=True)
    jid_abn, cid_abn, ts_abn = _make_ids(abn_trace, prefix="ta")
    df_abn = pd.concat([jid_abn, cid_abn, ts_abn, X_abn], axis=1)

    # normal
    norm_path = os.path.join(data_dir, TEST_NORM_FILE)
    df_norm_raw = _read_no_header_csv(norm_path)
    norm_trace, X_norm, _ = _split_features_and_label(df_norm_raw, has_label=False)
    jid_norm, cid_norm, ts_norm = _make_ids(norm_trace, prefix="tn")
    df_norm = pd.concat([jid_norm, cid_norm, ts_norm, X_norm], axis=1)

    # combine test data
    df_test = pd.concat([df_norm, df_abn], axis=0, ignore_index=True)
    df_test.to_hdf(OUT_TEST_HDF, key="prod_test_data", mode="w", format="table")
    print(f"Wrote {OUT_TEST_HDF}   shape={df_test.shape}")

    # labels
    labels_norm = pd.DataFrame({
        "job_id": jid_norm,
        "component_id": cid_norm,
        "binary_anom": 0
    })
    labels_abn = pd.DataFrame({
        "job_id": jid_abn,
        "component_id": cid_abn,
        "binary_anom": y_abn
    })
    df_labels = pd.concat([labels_norm, labels_abn], axis=0, ignore_index=True)

    df_labels["app_name"] = "eclipse"  # use Eclipse-like name to avoid filter drops
    df_labels["anom_name"] = df_labels["binary_anom"].map({0: "none", 1: "memleak"})

    df_labels.to_csv(OUT_TEST_LABELS, index=False)
    print(f"Wrote {OUT_TEST_LABELS}  shape={df_labels.shape}")

def main():
    build_train(DATA_DIR)
    build_test_and_labels(DATA_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
