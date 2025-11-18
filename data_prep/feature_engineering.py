import pandas as pd

def categorize_age(age):
    # 0: younger
    # 1:adults
    # 2: middle-aged
    if age < 30:
        return 0
    elif age < 45:
        return 1
    else:
        return 2

def data_prep(df):
    
    # Create time features
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    df_copy = df.copy()
    
    # Convert categorical features into numerical featurs
    product_ids_map = {'P001':1, 'P002':2, 'P003':3, 'P004':4,
                      'P005':5, 'P006':6, 'P007':7, 'P008':8}
    category_map = {'sports': 0, 'grocery': 1,'food': 2,
                    'clothing': 3, 'home': 4, 'electronics': 5, 'beauty': 6}
    region_map = {'Mumbai': 0, 'Delhi': 1, 'Coimbatore': 2,
                 'Chennai': 3, 'Salem': 4, 'Hyderabad': 5,
                 'Kochi': 6, 'Bangalore': 7}
    channel_map = {'retail': 0,'online': 1}

    df_copy['product_id'] = df_copy['product_id'].map(product_ids_map).astype('int')
    df_copy['category'] = df_copy['category'].map(category_map)
    df_copy['region'] = df_copy['region'].map(region_map)
    df_copy['channel'] = df_copy['channel'].map(channel_map)
    
    # Create a feature based on age-groups
    df_copy["age_group"] = df_copy["customer_age"].apply(categorize_age)
    
    # Create revenue feature
    df_copy["revenue"] = df_copy["price"]*df_copy["quantity"]
    
    df_copy = df_copy[["product_id", "year", "month", "day", "hour", "weekday", "is_weekend",
                       "category", "region", "customer_age", "age_group", "channel", "price", "quantity", "revenue"]]
    
    return df_copy




