import numpy as np


def make_decision(lr_model, rf_model, transaction):

    lr_prob = lr_model.predict_proba(transaction)[0][1]
    rf_prob = rf_model.predict_proba(transaction)[0][1]

    risk_score = np.mean([lr_prob, rf_prob])
    difference = abs(lr_prob - rf_prob)

    if risk_score < 0.30:
        decision = "APPROVE"
        reason = "Low fraud risk."

    elif risk_score > 0.75 and difference < 0.25:
        decision = "BLOCK"
        reason = "High fraud confidence."

    else:
        decision = "REVIEW"
        reason = "Model uncertainty detected."

    return risk_score, decision, reason
