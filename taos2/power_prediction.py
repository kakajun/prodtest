import taosws
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

# Configuration
DB_URL = "taosws://root:taosdata@192.168.2.110:6041"
DB_NAME = "station_data"
TABLE_NAME = "stable_gtjjlfgdzf"
START_DATE = "2026-01-22"
END_DATE = "2026-01-29" 

# Supervised Learning Target: Active Power (dc)
# Features: 
# b: Wind Speed
# c: Wind Direction
# a: Ambient Temp
# cd: Generator Speed
# f, g, h: Pitch Angles
FEATURE_KEYS = ['b', 'c', 'a', 'cd', 'f', 'g', 'h']
TARGET_KEY = 'dc'

def log(msg):
    print(msg)

def load_data(equ_code=None):
    log(f"Connecting to TDengine... (Device: {equ_code})")
    try:
        conn = taosws.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_NAME}")
        
        cols = ['ts', TARGET_KEY] + FEATURE_KEYS
        cols_str = ", ".join(cols)
        
        where_clause = f"ts >= '{START_DATE}' AND ts < '{END_DATE}'"
        if equ_code:
            where_clause += f" AND equ_code = '{equ_code}'"
            
        query = f"SELECT {cols_str} FROM {TABLE_NAME} WHERE {where_clause}"
        log(f"Executing query...")
        cursor.execute(query)
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=cols)
        
        # Process timestamp
        df['ts'] = pd.to_datetime(df['ts'])
        if df['ts'].dt.tz is not None:
            df['ts'] = df['ts'].dt.tz_localize(None)
            
        # Convert columns to numeric
        for col in [TARGET_KEY] + FEATURE_KEYS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except Exception as e:
        log(f"Error loading data: {e}")
        return None

def train_and_evaluate(df, equ_code):
    # Preprocessing
    df = df.dropna()
    
    # Simple outlier removal for physical feasibility
    # Power shouldn't be negative (unless consuming, but for prediction usually we focus on generation)
    # Wind speed shouldn't be negative
    df = df[df['b'] >= 0]
    
    X = df[FEATURE_KEYS]
    y = df[TARGET_KEY]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    log(f"Training RandomForest Regressor for {equ_code}...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    log(f"MSE: {mse:.2f}")
    log(f"R2 Score: {r2:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # 1. Actual vs Predicted Scatter
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Power (kW)')
    plt.ylabel('Predicted Power (kW)')
    plt.title(f'Actual vs Predicted ({equ_code})\nR2={r2:.3f}')
    
    # 2. Feature Importance
    plt.subplot(1, 2, 2)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [FEATURE_KEYS[i] for i in indices])
    plt.title("Feature Importance")
    
    plt.tight_layout()
    plt.savefig(f'Power_Prediction_{equ_code}.png')
    plt.close()
    
    return model

def get_equ_codes():
    try:
        conn = taosws.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_NAME}")
        cursor.execute(f"SELECT DISTINCT equ_code FROM {TABLE_NAME}")
        data = cursor.fetchall()
        conn.close()
        return sorted([row[0] for row in data])
    except:
        return []

def main():
    equ_codes = get_equ_codes()
    log(f"Found devices: {equ_codes}")
    
    # Train for first device as an example, or all
    # For demo purposes, let's just do the first one to save time
    if equ_codes:
        target_equ = equ_codes[0] # F01
        log(f"Running supervised learning example on {target_equ}")
        
        df = load_data(target_equ)
        if df is not None and not df.empty:
            train_and_evaluate(df, target_equ)
        else:
            log("No data found.")
    else:
        log("No devices found.")

if __name__ == "__main__":
    main()
