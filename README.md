UPI Fraud Detection System:

A machine learning-based fraud detection system designed to identify suspicious UPI transactions using cost-sensitive modeling and threshold optimization.
This project focuses on maximizing fraud capture (high recall) while maintaining practical decision boundaries similar to real-world payment risk engines.

Problem Statement:
Digital payment systems face significant financial and reputational risk due to fraudulent transactions. Traditional classification models often prioritize accuracy, which can result in missed fraud cases.
This project approaches fraud detection as a cost-sensitive machine learning problem, where false negatives are significantly more expensive than false positives.

Objectives:
- Detect fraudulent UPI transactions with high recall  
- Handle class imbalance effectively  
- Benchmark multiple ML models   

Models used:
- Logistic Regression  
- Random Forest  
- XGBoost (Benchmark Model)
- SMOTE
