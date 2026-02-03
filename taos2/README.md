# Supervised Learning: Wind Turbine Power Prediction

This project demonstrates a supervised learning approach using real data from TDengine.

## Objective
Predict the **Active Power Output** (`dc`) of a wind turbine based on environmental and operational features.

## Data Source
- **Database**: TDengine (`station_data.stable_gtjjlfgdzf`)
- **Features (Inputs)**:
  - `b`: Wind Speed (m/s)
  - `c`: Wind Direction (°)
  - `a`: Ambient Temperature (°C)
  - `cd`: Generator Speed (rpm)
  - `f`, `g`, `h`: Pitch Angles (°)
- **Target (Label)**:
  - `dc`: Active Power (kW)

## Methodology
1. **Data Loading**: Fetch time-series data for a specific device (e.g., F01).
2. **Preprocessing**: Handle missing values and filter invalid physical states.
3. **Model**: **Random Forest Regressor**.
   - A robust ensemble method suitable for non-linear relationships in wind turbine power curves.
4. **Evaluation**:
   - **R2 Score**: Measures how well the model explains the variance in power generation.
   - **MSE**: Mean Squared Error.
5. **Visualization**:
   - Scatter plot of Actual vs. Predicted Power.
   - Feature Importance chart (showing which sensors matter most).

## How to Run
```bash
python power_prediction.py
```

## Results
- The script generates `Power_Prediction_{equ_code}.png`.
- High R2 score (close to 1.0) indicates the turbine is following its expected power curve.
- Significant deviations could indicate potential issues (underperformance).
