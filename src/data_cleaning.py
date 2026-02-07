import pandas as pd
import yaml
import os

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

input_path = params["data"]["processed_path"]
output_path = "data/processed/final_data.csv"
target = params["data"]["target_column"]

def main():
    df = pd.read_csv(input_path)

    # Convert all columns except datetime to numeric
    for col in df.columns:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing target
    df.dropna(subset=[target], inplace=True)

    # Fill remaining missing values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Feature engineering
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    # Drop datetime after feature extraction
    df.drop(columns=["datetime"], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Data cleaning and feature engineering completed.")

if __name__ == "__main__":
    main()
