import pandas as pd
from pathlib import Path

def preprocess_blood_test_report(num_of_rows, df):
    df_sampled = df.sample(n=num_of_rows, random_state=42)
    df_sampled = df_sampled.reset_index(drop=True)
    df_sampled.drop([df_sampled.columns[0], 'Age', 'Gender'], axis=1, inplace=True)
    df_sampled['Diagnosis'] = df['Diagnosis'].replace({1: 'Diabetic', 0: 'Not Diabetic'})
    return df_sampled

def add_patient_id_col(df):
    df['patient_id'] = [f'p{str(i).zfill(3)}' for i in range(1, len(df) + 1)]
    
def save_csv(df, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath) 