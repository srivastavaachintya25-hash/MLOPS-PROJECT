import pandas as pd
import yaml
import os

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

raw_path = params["data"]["raw_path"]
processed_path = params["data"]["processed_path"]

date_col = params["features"]["datetime_column"]["date"]
time_col = params["features"]["datetime_column"]["time"]

def main():
    # Read raw data
    df = pd.read_csv(raw_path)

    # Create datetime column
    df["datetime"] = pd.to_datetime(
        df[date_col] + " " + df[time_col],
        dayfirst=True,
        errors="coerce"
    )

    # Drop original date & time columns
    df.drop(columns=[date_col, time_col], inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # Save processed data
    df.to_csv(processed_path, index=False)
    print("Data ingestion completed.")

if __name__ == "__main__":
    main()
