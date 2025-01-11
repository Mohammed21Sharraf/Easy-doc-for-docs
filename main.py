import pandas as pd
from pathlib import Path
from utilities.preprocess import preprocess_blood_test_report, add_patient_id_col

if __name__ == "__main__":
    blood_test_report_df = pd.read_csv('raw data/Blood Test Report.csv')
    blood_test_report_preprocessed = preprocess_blood_test_report(500, blood_test_report_df)
    add_patient_id_col(blood_test_report_preprocessed)

    filepath = Path('preprocessed data/Blood Test Report Preprocessed.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    blood_test_report_preprocessed.to_csv(filepath)  

    

