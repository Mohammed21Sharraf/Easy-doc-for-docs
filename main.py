import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from utilities.embed_xray_images import embed_images
from utilities.embedding import embed_numerical_features, combine_embeddings
from utilities.preprocess import preprocess_blood_test_report, add_patient_id_col, save_csv
from utilities.setup_qdrant import setup_qdrant, create_collection, upload_collection, upload_points

if __name__ == "__main__":
    folder_path = 'embeddings'
    required_files = ['blood_test_embeddings.npy', 'xray_embeddings.npy']
    missing_files = [file for file in required_files if not os.path.isfile(os.path.join(folder_path, file))]

    # Model used to tokenize categortical features
    model_BIOBert = "dmis-lab/biobert-v1.1"
    tokenizer_BIOBert = BertTokenizer.from_pretrained(model_BIOBert)
    BIOBert = BertModel.from_pretrained(model_BIOBert)

    # Model to embed X-ray Images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor_CLIP = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if missing_files:
        # Clean the dataset and save it to a folder
        blood_test_report_df = pd.read_csv('raw data/Blood Test Report.csv')
        blood_test_report_preprocessed = preprocess_blood_test_report(500, blood_test_report_df)
        add_patient_id_col(blood_test_report_preprocessed)
        save_csv(blood_test_report_preprocessed, 'preprocessed data/Blood Test Report Preprocessed.csv')

        # Create Embeddings and save embeddings
        blood_test_report_preprocessed, numeric_cols = embed_numerical_features(blood_test_report_preprocessed)
        blood_test_embeddings, blood_test_metadata = combine_embeddings(blood_test_report_preprocessed, numeric_cols, tokenizer_BIOBert, BIOBert)

        with open("metadata/blood_test_metadata.json", "w") as f:
            json.dump(blood_test_metadata, f, indent=4)
        np.save('embeddings/blood_test_embeddings.npy', blood_test_embeddings)
        print("BLOOD TEST REPORT EMBEDDING DONE")

        # Create X-ray Image Embedding
        xray_embeddings, xray_metadata = embed_images('raw data/X-ray Images', model_CLIP, processor_CLIP, device)

        with open("metadata/xray_metadata.json", "w") as f:
            json.dump(xray_metadata, f, indent=4)
        np.save('embeddings/xray_embeddings.npy', xray_embeddings)
        print("XRAY IMAGE EMBEDDING DONE")


    blood_test_embeddings = np.load('embeddings/blood_test_embeddings.npy', allow_pickle=True)
    with open("metadata/blood_test_metadata.json", "r") as f:
        blood_test_metadata = json.load(f)

    xray_embeddings = np.load('embeddings/xray_embeddings.npy', allow_pickle=True)
    with open('metadata/xray_metadata.json', 'r') as f:
        xray_metadata = json.load(f)

    client = setup_qdrant()

    create_collection(client, 'blood_tests', len(blood_test_embeddings[0]))
    create_collection(client, 'xray_images', len(xray_embeddings[0]))

    if client.count(collection_name='blood_tests').count == 0:
        upload_collection(client, 'blood_tests', blood_test_embeddings, blood_test_metadata)

    if client.count(collection_name='xray_images').count == 0:
        upload_points(client, 'xray_images', xray_embeddings, xray_metadata)
    

    

    




