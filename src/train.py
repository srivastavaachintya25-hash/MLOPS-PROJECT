import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_path = "data/processed/final_data.csv"
target = params["data"]["target_column"]

test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]

def main():
    df = pd.read_csv(data_path)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", test_size)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(model, "model")

        print("Training completed")
        print(f"MSE: {mse}")
        print(f"R2 Score: {r2}")

if __name__ == "__main__":
    main()
