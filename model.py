# model.py
import os
import requests
import bz2
from tqdm import tqdm

def download_models():
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs and file paths
    models = {
        "shape_predictor": {
            "url": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
            "compressed": os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat.bz2"),
            "decompressed": os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
        },
        "face_recognition": {
            "url": "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2",
            "compressed": os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat.bz2"),
            "decompressed": os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
        }
    }
    
    for model_name, paths in models.items():
        # Skip if the decompressed file already exists
        if os.path.exists(paths["decompressed"]):
            print(f"{model_name} model already exists.")
            continue
            
        print(f"\nDownloading {model_name} model...")
        
        try:
            # Download the file
            response = requests.get(paths["url"], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Show progress bar while downloading
            with open(paths["compressed"], 'wb') as file, tqdm(
                desc=paths["compressed"],
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
            
            print(f"Decompressing {model_name} model...")
            # Decompress the file
            with open(paths["decompressed"], 'wb') as new_file:
                with open(paths["compressed"], 'rb') as file:
                    data = bz2.decompress(file.read())
                    new_file.write(data)
            
            # Remove the compressed file
            os.remove(paths["compressed"])
            print(f"{model_name} model successfully downloaded and decompressed.")
            
        except Exception as e:
            print(f"Error downloading {model_name} model: {str(e)}")
            # Clean up any partially downloaded files
            for file_path in [paths["compressed"], paths["decompressed"]]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            return False
    
    return True

if __name__ == "__main__":
    print("Starting model download...")
    try:
        if download_models():
            print("\nAll models successfully downloaded and installed!")
            print("Models are saved in the 'models' directory.")
        else:
            print("\nError downloading models. Please try again.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")


