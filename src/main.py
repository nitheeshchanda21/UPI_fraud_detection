import joblib

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features
from src.train_models import train_models
from src.decision_engine import make_decision


def main():

    print("Loading dataset...")
    df = load_data("data/upi_fraud.csv")
    print(df.columns)


    print("Performing feature engineering...")
    df = create_features(df)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training models...")
    logistic, random_forest, xgboost = train_models(X_train, y_train)

    from src.train_models import evaluate
    evaluate(logistic, X_test, y_test, "Logistic Regression")
    evaluate(random_forest, X_test, y_test, "Random Forest")
    evaluate(xgboost, X_test, y_test, "XGBoost")


    print("Saving Random Forest model...")
    joblib.dump(random_forest, "models/random_forest.pkl")

    print("Running decision engine on a sample transaction...")

    sample = X_test.iloc[[0]]

    risk, decision, reason = make_decision(
        logistic,
        random_forest,
        sample
    )

    print("\n===== FRAUD RISK OUTPUT =====")
    print(f"Risk Score: {risk:.2f}")
    print(f"Decision: {decision}")
    print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
