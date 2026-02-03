import taosws
import pandas as pd
import numpy as np

# Configuration
DB_URL = "taosws://root:taosdata@192.168.2.110:6041"
DB_NAME = "station_data"
TABLE_NAME = "stable_gtjjlfgdzf"
START_DATE = "2026-01-22"
END_DATE = "2026-01-29"
TRAIN_END_DATE = "2026-01-27"

FAULT_KEYS = [
    'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'ha', 'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hh', 'hi', 'hj',
    'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs', 'ht', 'hu', 'hv', 'hw', 'hx'
]
EXTRA_FAULT_KEYS = ['db', 'gt']

def main():
    print("Connecting to TDengine...")
    conn = taosws.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute(f"USE {DB_NAME}")

    cols = FAULT_KEYS + EXTRA_FAULT_KEYS
    cols_str = ", ".join([f"SUM({c})" for c in cols])

    # Check total rows
    query_total = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE ts >= '{START_DATE}' AND ts < '{END_DATE}'"
    cursor.execute(query_total)
    total_rows = cursor.fetchone()[0]
    print(f"Total rows: {total_rows}")

    # Check rows with at least one fault
    # In TDengine SQL, we can't easily do WHERE (c1 + c2 + ...) > 0 efficiently without fetching.
    # But we can select the columns and process in pandas for a sample or counts if small.
    # Actually, let's just fetch the fault columns.

    print("Fetching fault columns...")
    cols_select = ", ".join(cols)
    query = f"SELECT ts, {cols_select} FROM {TABLE_NAME} WHERE ts >= '{START_DATE}' AND ts < '{END_DATE}'"
    cursor.execute(query)
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=['ts'] + cols)
    df['ts'] = pd.to_datetime(df['ts'])
    if df['ts'].dt.tz is not None:
        df['ts'] = df['ts'].dt.tz_localize(None)

    df['is_fault'] = df[cols].sum(axis=1) > 0

    train_mask = df['ts'] < pd.to_datetime(TRAIN_END_DATE)
    test_mask = ~train_mask

    df_train = df[train_mask]
    df_test = df[test_mask]

    print(f"\nTraining Set ({START_DATE} to {TRAIN_END_DATE}):")
    print(f"Total samples: {len(df_train)}")
    print(f"Faulty samples: {df_train['is_fault'].sum()}")
    print(f"Fault ratio: {df_train['is_fault'].mean():.2%}")

    print(f"\nTest Set ({TRAIN_END_DATE} to {END_DATE}):")
    print(f"Total samples: {len(df_test)}")
    print(f"Faulty samples: {df_test['is_fault'].sum()}")
    print(f"Fault ratio: {df_test['is_fault'].mean():.2%}")

    # Which faults are most common?
    fault_counts = df[cols].sum().sort_values(ascending=False)
    print("\nTop 5 Faults:")
    print(fault_counts.head())

if __name__ == "__main__":
    main()
