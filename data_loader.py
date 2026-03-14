import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path=r"D:\\NEW_VOLUME_E\\ML_PROJECTS\\UNSUPERVISED_PROJECTS\\FRAUD_DETECTION\\data\\creditcard.csv"):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    
    df = df.drop(['Time', 'Amount'], axis=1)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Shape: {X.shape} | Fraud rate: {y.mean():.5f} ({y.sum()} frauds)")
    return X, y