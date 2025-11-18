## System Design

**Simple workflow**:

Raw Data -> PySpark ETL -> Feature Store -> Model Training -> Model Registry ->  Docker Containerization -> FastAPI  -> Monitoring

**Key components**:
1. **Raw data**: Collect raw data from multiple sources.
2. **PySpark ETL**: PySpark is capable of handling large-scale data (5M+ rows/day). In this stage various data processing steps are performed:
    - Removing missing/duplicate/anomalous values
    - Converting data types
    - Feature transformations
    - Scaling to millions of rows using distributed processing across multiple machines
3. **Feature Store**: Store engineered features that are to be utlized for model training.
4. **Model Training**: Train your models based on features pulled from the feature store.
    - Regression: Random Forest, XGBoost, Linear Regression
    - Classification: Logistic Regression, XGBoost, Random Forest
    - Time-series forecasting: ARIMA, LSTM
5. **Model Registry**: Version, manage, track, and approve models before they are deployed into production.
6. **Docker Containerization**: After the model is approved in the model registry, one can containerize the complete application into a deployable Docker image.
7. **FastAPI (Prediction API)**: This is your deployment layer.     
    - FastAPI exposes endpoints for predicting revenue, channel, forecasting sales or revenue, etc. 
    - The latest approved model is loaded from the Model Registry.
8. **Monitoring**: Automate continuous monitoring of data drift, model drift, system performance, and trigger alerts when needed. We can set email notifications for anomalies detected, so that the team can take actions quickly.
    - Data drift: 
        - Occurs when input data distribution changes from what the model was trained on.
        - For each feature one can perform statistical tests and compute drift score. 
        - Alerts can be triggered when drift score>threshold.
    - Model drift:
        - Occurs when the relationship between inputs and output changes.
        - One can monitor prediction error metrics such as R² score, Accuracy, F1, MAE, RMSE over time.
        - One can trigger alerts when there is a sudden drop in R² score, Accuracy, F1 and sudden increase in MAE, RMSE.
        - One can compare current  model performance vs training/reference performance and trigger alert when there is a sudden drift in both.
    - Anomaly detection:
        - Conducts checks for unusual or unexpected behavior in real-time (e.g., sudden spikes in price, quantity, transactions, etc.).
        - One can use statistical outlier detection methods such as IQR and Z-score.
            - IQR: Any value outside the lower (Q1 - 1.5IQR) and upper (Q3 + 1.5IQR) limits is marked as an anomaly and alerts can be triggered.
            - Z-score: Values with |z_score|>3 is termed as anomaly and alerts can be triggered