import pandas as pd
from utilities.preprocess import preprocess_blood_test_report, add_patient_id_col, save_csv
from utilities.embedding import embed_numerical_features, combine_embeddings
from transformers import BertTokenizer, BertModel
import torch

if __name__ == "__main__":
    # Model used to tokenize categortical features
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Clean the dataset and save it to a folder
    blood_test_report_df = pd.read_csv('raw data/Blood Test Report.csv')
    blood_test_report_preprocessed = preprocess_blood_test_report(500, blood_test_report_df)
    add_patient_id_col(blood_test_report_preprocessed)
    save_csv(blood_test_report_preprocessed, 'preprocessed data/Blood Test Report Preprocessed.csv')

    # Create Embeddings
    blood_test_report_preprocessed, numeric_cols = embed_numerical_features(blood_test_report_preprocessed)
    blood_test_embeddings = combine_embeddings(blood_test_report_preprocessed, numeric_cols, tokenizer, model)
    print(blood_test_embeddings[0])


