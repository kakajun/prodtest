import taosws
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
DB_URL = "taosws://root:taosdata@192.168.2.110:6041"
DB_NAME = "station_data"
TABLE_NAME = "stable_gtjjlfgdzf"
START_DATE = "2026-01-22"
END_DATE = "2026-01-29"
EQU_CODE = "F02"

def main():
    print(f"Connecting to TDengine to fetch data for {EQU_CODE}...")
    try:
        conn = taosws.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_NAME}")

        # Select ts, wind speed (b), and active power (dc)
        query = f"SELECT ts, b, dc FROM {TABLE_NAME} WHERE ts >= '{START_DATE}' AND ts < '{END_DATE}' AND equ_code = '{EQU_CODE}'"
        print(f"Executing query: {query}")
        cursor.execute(query)
        data = cursor.fetchall()

        if not data:
            print("No data found.")
            return

        df = pd.DataFrame(data, columns=['ts', 'wind_speed', 'power'])

        # Process timestamp
        df['ts'] = pd.to_datetime(df['ts'])
        if df['ts'].dt.tz is not None:
            df['ts'] = df['ts'].dt.tz_localize(None)

        # Sort by time
        df = df.sort_values('ts')

        print(f"Data fetched: {len(df)} rows")

        # Plotting
        plt.figure(figsize=(15, 8))

        # Create primary axis for Wind Speed
        ax1 = plt.gca()
        line1 = ax1.plot(df['ts'], df['wind_speed'], color='blue', label='Wind Speed (m/s)', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Wind Speed (m/s)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)

        # Create secondary axis for Power
        ax2 = ax1.twinx()
        line2 = ax2.plot(df['ts'], df['power'], color='green', label='Active Power (kW)', alpha=0.7, linewidth=1)
        ax2.set_ylabel('Active Power (kW)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title(f'Wind Speed vs Active Power for {EQU_CODE} ({START_DATE} to {END_DATE})')
        plt.tight_layout()

        output_file = f'{EQU_CODE}_wind_power.png'
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
