import taosws
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import gc
import joblib
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

# Configuration
DB_URL = "taosws://root:taosdata@192.168.2.110:6041"
DB_NAME = "station_data"
TABLE_NAME = "stable_gtjjlfgdzf"
START_DATE = "2026-01-22"
END_DATE = "2026-01-29"
TRAIN_END_DATE = "2026-01-27"
MAPPING_FILE = "column_mapping.json"

FEATURE_KEYS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al',
    'am', 'an', 'ao', 'ap', 'aq', 'ar', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd',
    'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu',
    'bv', 'bw', 'bx', 'bz', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm',
    'cn', 'co', 'cp', 'cq', 'cr', 'cs', 'ct', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'da'
]

FAULT_KEYS = [
    'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'ha', 'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hh', 'hi', 'hj',
    'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs', 'ht', 'hu', 'hv', 'hw', 'hx'
]

EXTRA_FAULT_KEYS = ['db', 'gt']

def log(msg):
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg)

def get_column_mapping():
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data(equ_code=None):
    log(f"Connecting to TDengine... (Device: {equ_code})")
    try:
        conn = taosws.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_NAME}")

        # Construct query with specific columns
        cols_to_select = ['ts'] + FEATURE_KEYS + FAULT_KEYS + EXTRA_FAULT_KEYS
        cols_to_select = list(dict.fromkeys(cols_to_select)) # Remove duplicates
        cols_str = ", ".join(cols_to_select)

        where_clause = f"ts >= '{START_DATE}' AND ts < '{END_DATE}'"
        if equ_code:
            where_clause += f" AND equ_code = '{equ_code}'"

        query = f"SELECT {cols_str} FROM {TABLE_NAME} WHERE {where_clause}"
        log(f"Executing query: {query}")
        cursor.execute(query)
        data = cursor.fetchall()
        log(f"Rows fetched: {len(data)}")

        if not data:
            return None

        # Optimize memory by creating DataFrame directly with float32 for numeric cols if possible
        # But 'data' is mixed (ts is string/datetime, others float).
        # We create DF first.
        log("Creating DataFrame...")
        df = pd.DataFrame(data, columns=cols_to_select)
        del data
        gc.collect()

        log(f"Data loaded: {df.shape}")

        # Optimize types
        cols = df.columns
        for col in cols:
            if col != 'ts':
                df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    except Exception as e:
        log(f"Error loading data: {e}")
        import traceback
        log(traceback.format_exc())
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def preprocess_data(df, mapping):
    log("Preprocessing data...")
    rename_dict = {}
    for k, v in mapping.items():
        if k in df.columns:
            rename_dict[k] = v

    df_renamed = df.rename(columns=rename_dict)

    if 'ts' in df_renamed.columns:
        df_renamed['ts'] = pd.to_datetime(df_renamed['ts'])
        if df_renamed['ts'].dt.tz is not None:
            df_renamed['ts'] = df_renamed['ts'].dt.tz_localize(None)
        df_renamed = df_renamed.set_index('ts')

    feature_cols = [mapping[k] for k in FEATURE_KEYS if k in mapping and mapping[k] in df_renamed.columns]
    fault_cols = [mapping[k] for k in FAULT_KEYS if k in mapping and mapping[k] in df_renamed.columns]
    extra_fault_cols = [mapping[k] for k in EXTRA_FAULT_KEYS if k in mapping and mapping[k] in df_renamed.columns]

    X = df_renamed[feature_cols].ffill().bfill().fillna(0)

    all_fault_cols = fault_cols + extra_fault_cols

    y_ground_truth = df_renamed[all_fault_cols].sum(axis=1) > 0
    y_ground_truth = y_ground_truth.astype(int)

    return df_renamed, X, y_ground_truth

def train_predict_plot(model_name, model, X_train, X_test, y_test_gt, train_index, test_index, equ_code):
    log(f"\nTraining {model_name} for {equ_code}...")

    model.fit(X_train)

    # PyOD might return float64 scores.
    # decision_scores_ is a property.
    train_scores = model.decision_scores_
    test_scores = model.decision_function(X_test)
    threshold = model.threshold_
    log(f"Threshold: {threshold}")

    plt.figure(figsize=(15, 6))
    plt.plot(test_index, test_scores, label='Anomaly Score', color='blue', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

    anom_indices = y_test_gt[y_test_gt == 1].index
    if not anom_indices.empty:
        plt.scatter(anom_indices, [max(test_scores)] * len(anom_indices),
                   color='red', marker='x', s=20, label='Actual Faults', zorder=5)

    plt.title(f'{model_name} Anomaly Detection (Test Set: Jan 27-28) - Device {equ_code}')
    plt.ylabel('Anomaly Score')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_results_{equ_code}.png')
    log(f"Saved plot to {model_name}_results_{equ_code}.png")
    plt.close()

    # Save model
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"{model_name}_{equ_code}.joblib")
    joblib.dump(model, model_path)
    log(f"Saved model to {model_path}")

def get_equ_codes():
    log("Fetching distinct equ_codes...")
    try:
        conn = taosws.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_NAME}")
        cursor.execute(f"SELECT DISTINCT equ_code FROM {TABLE_NAME}")
        data = cursor.fetchall()
        conn.close()
        # data is list of tuples like [('F15',), ('F24',), ...]
        codes = [row[0] for row in data]
        return sorted(codes)
    except Exception as e:
        log(f"Error fetching equ_codes: {e}")
        return []

def main():
    if os.path.exists("debug.log"):
        os.remove("debug.log")

    mapping = get_column_mapping()
    equ_codes = get_equ_codes()
    log(f"Found devices: {equ_codes}")

    # Process each device
    for equ_code in equ_codes:
        log(f"\n{'='*50}\nProcessing Device: {equ_code}\n{'='*50}")

        df = load_data(equ_code)
        if df is None or df.empty:
            log(f"No data found for {equ_code}. Skipping.")
            continue

        # Check if we have enough data for split
        if len(df) < 100:
            log(f"Not enough data for {equ_code} ({len(df)} rows). Skipping.")
            continue

        df_full, X, y_gt = preprocess_data(df, mapping)

        train_mask = df_full.index < pd.to_datetime(TRAIN_END_DATE)
        test_mask = ~train_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_test_gt = y_gt[test_mask]

        if X_train.empty or X_test.empty:
            log(f"Train or Test set empty for {equ_code}. Skipping.")
            continue

        log(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            ecod = ECOD(contamination=0.01)
            train_predict_plot("ECOD", ecod, X_train_scaled, X_test_scaled, y_test_gt, X_train.index, X_test.index, equ_code)

            iforest = IForest(contamination=0.01, n_jobs=-1, random_state=42)
            train_predict_plot("IForest", iforest, X_train_scaled, X_test_scaled, y_test_gt, X_train.index, X_test.index, equ_code)
        except Exception as e:
            log(f"Error training model for {equ_code}: {e}")
            import traceback
            log(traceback.format_exc())

    log("\nAll Done.")

if __name__ == "__main__":
    main()
