import pandas as pd
from utilities.preprocess import preprocess_blood_test_report, add_patient_id_col, save_csv
from utilities.embedding import embed_numerical_features, combine_embeddings
from utilities.embed_xray_images import embed_images
from transformers import BertTokenizer, BertModel
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

if __name__ == "__main__":
    # Model used to tokenize categortical features
    model_BIOBert = "dmis-lab/biobert-v1.1"
    tokenizer_BIOBert = BertTokenizer.from_pretrained(model_BIOBert)
    BIOBert = BertModel.from_pretrained(model_BIOBert)

    # Model to embed X-ray Images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor_CLIP = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Clean the dataset and save it to a folder
    blood_test_report_df = pd.read_csv('raw data/Blood Test Report.csv')
    blood_test_report_preprocessed = preprocess_blood_test_report(500, blood_test_report_df)
    add_patient_id_col(blood_test_report_preprocessed)
    save_csv(blood_test_report_preprocessed, 'preprocessed data/Blood Test Report Preprocessed.csv')

    # Create Embeddings and save embeddings
    blood_test_report_preprocessed, numeric_cols = embed_numerical_features(blood_test_report_preprocessed)
    blood_test_embeddings = combine_embeddings(blood_test_report_preprocessed, numeric_cols, tokenizer_BIOBert, BIOBert)
    np.save('embeddings/blood_test_embeddings.npy', blood_test_embeddings)
    print("BLOOD TEST REPORT EMBEDDING DONE")

    # Create X-ray Image Embedding
    xray_embeddings, xray_metadata = embed_images('raw data/X-ray Images', model_CLIP, processor_CLIP, device)
    print(xray_embeddings[0])
    np.save('embeddings/xray_embeddings.npy', xray_embeddings)
    print("XRAY IMAGE EMBEDDING DONE")




