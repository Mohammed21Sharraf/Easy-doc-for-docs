import os

def rename_xray_files(directory):
    """
    Rename chest X-ray files based on category and assign patient IDs.
    
    Args:
    - directory (str): Path to the directory containing the files.
    """
    patient_id = 1  # Start patient ID counter

    for file_name in os.listdir(directory):
        # Skip non-image files
        if not file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
            continue

        # Determine category based on file name
        category = None
        if "COVID19" in file_name:
            category = "covid"
        elif file_name.startswith("IM-"):
            category = "normal"
        elif "bacteria" in file_name.lower() or "virus" in file_name.lower():
            category = "pneumonia"
        elif "Tuberculosis" in file_name:
            category = "tuberculosis"

        if category is None:
            print(f"Skipping {file_name} (unknown category)")
            continue

        # Generate new file name
        patient_id_str = f"p{patient_id:03d}"  # Format patient ID as p001, p002, etc.
        _, ext = os.path.splitext(file_name)  # Preserve original file extension
        new_file_name = f"{patient_id_str}_{category}{ext}"

        # Rename the file
        old_file_path = os.path.join(directory, file_name)
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(old_file_path, new_file_path)

        print(f"Renamed: {file_name} -> {new_file_name}")
        patient_id += 1  # Increment patient ID counter

# Example usage
directory = "raw data/X-ray Images"
rename_xray_files(directory)
