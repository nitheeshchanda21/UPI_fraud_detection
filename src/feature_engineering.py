def create_features(df):

    # Transaction deviation feature
    if "TransactionAmount" in df.columns and "AvgTransactionAmount" in df.columns:
        df["Deviation"] = df["TransactionAmount"] / (df["AvgTransactionAmount"] + 1)

    return df
