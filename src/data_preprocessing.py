import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # DROP columns that should NEVER be features
    columns_to_drop = [
        "TransactionID",
        "UserID",
        "DeviceID",
        "IPAddress",
        "PhoneNumber",
        "Timestamp",
        "Latitude",
        "Longitude"
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Convert categorical → numeric
    df = pd.get_dummies(df, drop_first=True)

    target_column = "FraudFlag"

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test
