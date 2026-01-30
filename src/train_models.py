from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(X_train, y_train):

    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    logistic = LogisticRegression(
        max_iter=3000,
        class_weight='balanced'
    )


    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    xgboost = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=fraud_ratio
    )


    logistic.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    xgboost.fit(X_train, y_train)

    return logistic, random_forest, xgboost
from sklearn.metrics import recall_score, f1_score

def evaluate(model, X_test, y_test, name):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.07).astype(int)


    print(f"\n{name} Performance:")
    print("Recall:", round(recall_score(y_test, preds), 3))
    print("F1 Score:", round(f1_score(y_test, preds), 3))

