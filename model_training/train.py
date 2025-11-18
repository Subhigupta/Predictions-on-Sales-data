from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split


def rf_regressor(df):
    
    x_df = df[[ "channel", "quantity", "category", "region", "day" , 
               "is_weekend", "weekday", "month", "hour", "customer_age" , "age_group", "product_id"]]
    y_df = df[["price"]]
    
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
    
    test_indices = x_test.index
    train_indices = x_train.index
    
    print("Fitting Random Forest Regressor....")
    rf_regr = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    rf_regr.fit(x_train, y_train)
    
    price_pred = rf_regr.predict(x_test)
    quantity_test = df.loc[test_indices, 'quantity'].values
    revenue_pred = price_pred * quantity_test
    revenue_actual = df.loc[test_indices, 'revenue'].values
    
    score = r2_score(revenue_actual, revenue_pred)
    mae = mean_absolute_error(revenue_actual, revenue_pred)
    rmse = root_mean_squared_error(revenue_actual, revenue_pred)
    
    print("Evaluation metric over test dataset....")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)
    
    price_pred = rf_regr.predict(x_train)
    quantity_train = df.loc[train_indices, 'quantity'].values
    revenue_pred = price_pred * quantity_train
    revenue_actual = df.loc[train_indices, 'revenue'].values
    
    score = r2_score(revenue_actual, revenue_pred)
    mae = mean_absolute_error(revenue_actual, revenue_pred)
    rmse = root_mean_squared_error(revenue_actual, revenue_pred)
    
    print("Evaluation metric over train dataset....")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)

def xgb_regressor(df):
    
    x_df = df[[ "channel", "quantity", "category", "region", "day" , 
               "is_weekend", "weekday", "month", "hour", "customer_age" , "age_group", "product_id"]]
    y_df = df[["price"]]
    
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
    
    test_indices = x_test.index
    train_indices = x_train.index
    
    print("Fitting XGBoost Regressor....")
    xgb_regr = XGBRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    xgb_regr.fit(x_train, y_train)
    
    price_pred = xgb_regr.predict(x_test)
    quantity_test = df.loc[test_indices, 'quantity'].values
    revenue_pred = price_pred * quantity_test
    revenue_actual = df.loc[test_indices, 'revenue'].values
    
    score = r2_score(revenue_actual, revenue_pred)
    mae = mean_absolute_error(revenue_actual, revenue_pred)
    rmse = root_mean_squared_error(revenue_actual, revenue_pred)
    
    print("Evaluation metric over test dataset....")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)
    
    price_pred = xgb_regr.predict(x_train)
    quantity_train = df.loc[train_indices, 'quantity'].values
    revenue_pred = price_pred * quantity_train
    revenue_actual = df.loc[train_indices, 'revenue'].values
    
    score = r2_score(revenue_actual, revenue_pred)
    mae = mean_absolute_error(revenue_actual, revenue_pred)
    rmse = root_mean_squared_error(revenue_actual, revenue_pred)
    
    print("Evaluation metric over train dataset....")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)

def rf_classifier(df):
    
    x_df = df[["quantity", "category", "region", "day" , "price",
               "is_weekend", "weekday", "month", "hour", "customer_age" , "age_group", "product_id"]]
    y_df = df[["channel"]]
    
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
    
    test_indices = x_test.index
    train_indices = x_train.index
    
    print("Fitting Random Forest Classifier....")
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    rf_clf.fit(x_train, y_train)
    
    channel_pred = rf_clf.predict(x_test)
    
    print("Evaluation metrics over Test dataset....")
    print("Accuracy:", accuracy_score(y_test, channel_pred))
    print("F1 Score:", f1_score(y_test, channel_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, channel_pred))

    channel_pred = rf_clf.predict(x_train)
    
    print("Evaluation metrics over Train dataset....")
    print("Accuracy:", accuracy_score(y_train, channel_pred))
    print("F1 Score:", f1_score(y_train, channel_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_train, channel_pred))
    
def xgb_classifier(df):
    
    x_df = df[["quantity", "category", "region", "day" , "price",
               "is_weekend", "weekday", "month", "hour", "customer_age" , "age_group", "product_id"]]
    y_df = df[["channel"]]
    
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
    
    test_indices = x_test.index
    train_indices = x_train.index
    
    print("Fitting XGBoost Classifier....")
    xgb_clf = XGBClassifier(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    xgb_clf.fit(x_train, y_train)
    
    channel_pred = xgb_clf.predict(x_test)
    
    print("Evaluation metrics over Test dataset....")
    print("Accuracy:", accuracy_score(y_test, channel_pred))
    print("F1 Score:", f1_score(y_test, channel_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, channel_pred))

    channel_pred = xgb_clf.predict(x_train)
    
    print("Evaluation metrics over Train dataset....")
    print("Accuracy:", accuracy_score(y_train, channel_pred))
    print("F1 Score:", f1_score(y_train, channel_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_train, channel_pred))
