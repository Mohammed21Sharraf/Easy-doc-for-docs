import os
import numpy
import torch
from PIL import Image
from torchvision import transforms

def embed_images(directory, model, processor, device):
    embeddings = []
    metadata = []

    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for file_name in os.listdir(directory):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        # Load and preprocess the image
        file_path = os.path.join(directory, file_name)
        image = Image.open(file_path).convert('RGB')  # Ensure RGB format
        image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

        # Extract embedding
        with torch.no_grad():
            inputs = processor(images=image_tensor, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            embedding = outputs.cpu().numpy().flatten()  # Convert to 1D array
        
        embeddings.append(embedding)
        metadata.append({'file_name': file_name, 'file_path': file_path})

        print(f"Processed: {file_name}")

    return embeddings, metadata