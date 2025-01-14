import numpy as np
from sklearn.preprocessing import StandardScaler

def embed_numerical_features(df):  
    scalar = StandardScaler()
    numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
    df[numeric_cols] = scalar.fit_transform(df[numeric_cols])
    return df, numeric_cols

def embed_categorical_features(text, tokenizer, model):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    # Get the embeddings for the `[CLS]` token (start token)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

def combine_embeddings(df, numeric_cols, tokenizer, model):
    combined_embeddings = []
    metadata = []
    
    for index, row in df.iterrows():
        numeric_embedding = row[numeric_cols].values
        text_embedding = embed_categorical_features(row['Diagnosis'], tokenizer, model)
        patient_id = row['patient_id']
        
        combined_embedding = np.concatenate((numeric_embedding, text_embedding.flatten()))
        combined_embeddings.append(combined_embedding)
        metadata.append({'patient_id': patient_id})

    return np.array(combined_embeddings), metadata

    


