import cv2
import dlib
import numpy as np
import pickle
import os
from scipy.spatial.distance import cosine
import requests
import bz2
from tqdm import tqdm

class FaceRecognitionSystem:
    def __init__(self, models_dir=None):
        # Set up model directory
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Define model paths
        self.shape_predictor_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
        self.face_recognition_model_path = os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
        self.database_path = os.path.join(models_dir, "face_database.pkl")
        
        # Detection range parameters
        self.min_face_size = 150  # Minimum face width in pixels
        self.max_face_size = 250  # Maximum face width in pixels
        self.detection_range_color = {
            'too_far': (0, 0, 255),    # Red
            'too_close': (0, 0, 255),  # Red
            'good': (0, 255, 0)        # Green
        }
        
        # Download models if needed
        self.download_models()
        
        # Check model files
        self._check_model_files()
        
        try:
            # Initialize face detector and models
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)
            self.face_encoder = dlib.face_recognition_model_v1(self.face_recognition_model_path)
            
            # Load or create face database
            self.face_database = self.load_database()
            
            # Recognition threshold
            self.recognition_threshold = 0.8
            
            print("Face Recognition System initialized successfully!")
            
        except Exception as e:
            print(f"\nError initializing Face Recognition System: {str(e)}")
            print("\nPlease ensure:")
            print("1. You have downloaded both .dat files")
            print("2. The files are placed in the 'models' folder")
            print("3. You have installed all required packages:")
            print("   pip install dlib opencv-python numpy scipy requests tqdm")
            raise

    def download_models(self):
        """Download required model files if they don't exist"""
        models = {
            "shape_predictor": {
                "url": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
                "compressed": self.shape_predictor_path + ".bz2",
                "decompressed": self.shape_predictor_path
            },
            "face_recognition": {
                "url": "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2",
                "compressed": self.face_recognition_model_path + ".bz2",
                "decompressed": self.face_recognition_model_path
            }
        }

        for model_name, paths in models.items():
            if os.path.exists(paths["decompressed"]):
                continue

            print(f"\nDownloading {model_name} model...")
            try:
                # Download file
                response = requests.get(paths["url"], stream=True)
                total_size = int(response.headers.get('content-length', 0))

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

                # Decompress file
                print(f"Decompressing {model_name} model...")
                with open(paths["decompressed"], 'wb') as new_file:
                    with open(paths["compressed"], 'rb') as file:
                        data = bz2.decompress(file.read())
                        new_file.write(data)

                # Remove compressed file
                os.remove(paths["compressed"])
                print(f"{model_name} model successfully downloaded and decompressed.")

            except Exception as e:
                print(f"Error downloading {model_name} model: {str(e)}")
                # Clean up partial downloads
                for file_path in [paths["compressed"], paths["decompressed"]]:
                    if os.path.exists(file_path):
                        os.remove(file_path)

    def _check_model_files(self):
        """Check if required model files exist"""
        missing_files = []
        
        for path in [self.shape_predictor_path, self.face_recognition_model_path]:
            if not os.path.exists(path):
                missing_files.append(os.path.basename(path))
        
        if missing_files:
            print("\nError: Missing required model files:")
            for file in missing_files:
                print(f"- {file}")
            raise FileNotFoundError("Missing required model files")

    def load_database(self):
        """Load or create face database"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading database: {str(e)}")
                return {'embeddings': [], 'names': []}
        return {'embeddings': [], 'names': []}

    def save_database(self):
        """Save face database"""
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.face_database, f)
        except Exception as e:
            print(f"Error saving database: {str(e)}")

    def get_face_embedding(self, frame, face):
        """Generate face embedding"""
        shape = self.shape_predictor(frame, face)
        return np.array(self.face_encoder.compute_face_descriptor(frame, shape))

    def recognize_face(self, face_embedding):
        """Recognize a face from its embedding"""
        if not self.face_database['embeddings']:
            return None
        
        distances = [cosine(face_embedding, known_embedding) 
                    for known_embedding in self.face_database['embeddings']]
        
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        if min_distance < self.recognition_threshold:
            return self.face_database['names'][min_distance_idx]
        return None

    def check_detection_range(self, face_width):
        """Check if face is within the desired detection range"""
        if face_width < self.min_face_size:
            return 'too_far'
        elif face_width > self.max_face_size:
            return 'too_close'
        return 'good'

    def draw_detection_info(self, frame, face, range_status):
        """Draw detection box and range information"""
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_width = x2 - x1
        
        # Get color based on range status
        color = self.detection_range_color[range_status]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw range status text
        if range_status == 'too_far':
            message = "Move Closer"
        elif range_status == 'too_close':
            message = "Move Back"
        else:
            message = "Good Distance"
            
        cv2.putText(frame, message, (x1, y2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw distance indicator
        cv2.putText(frame, f"Distance: {face_width}px", (x1, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def add_new_face(self, frame, face, name):
        """Add a new face to the database"""
        try:
            face_embedding = self.get_face_embedding(frame, face)
            self.face_database['embeddings'].append(face_embedding)
            self.face_database['names'].append(name)
            self.save_database()
            print(f"Successfully added new face for {name}")
            return True
        except Exception as e:
            print(f"Error adding new face: {str(e)}")
            return False

    def start_recognition(self):
        """Start detection and recognition"""
        try:
            name = "Unknown"
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")
            
            print("\nPress 'q' to quit recognition")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                faces = self.detector(rgb_frame)

                for face in faces:
                    # Check face distance range
                    face_width = face.right() - face.left()
                    range_status = self.check_detection_range(face_width)

                    # Draw detection box and range information
                    self.draw_detection_info(frame, face, range_status)

                    # Only process face if it's in good range
                    if range_status == 'good':
                        # Recognize face
                        face_embedding = self.get_face_embedding(rgb_frame, face)
                        name = self.recognize_face(face_embedding)

                        label = name if name else "Unknown"
                        cv2.putText(frame, label, (face.left(), face.top() - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imshow('Face Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                if name not in ["Unknown", None]:
                    return name
                
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error during recognition: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def add_person(self):
        """Add a new person to the database"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")

            print("\nPosition the person in front of the camera. The system will automatically save the face when detected.")
            print("Press 'q' to cancel.")

            name = input("Enter the name of the new person: ")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                faces = self.detector(rgb_frame)

                for face in faces:
                    # Check face distance range
                    face_width = face.right() - face.left()
                    range_status = self.check_detection_range(face_width)

                    # Draw detection box and range information
                    self.draw_detection_info(frame, face, range_status)

                    if range_status == 'good':
                        if self.add_new_face(rgb_frame, face, name):
                            print(f"Face for {name} added successfully!")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        else:
                            print("Failed to add the face. Try again.")

                cv2.imshow('Add New Person', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Canceled adding new person.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error adding new person: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
         
        
    def menu(self):
        """Main menu for the system"""
        while True:
            print("\nSelect an option:")
            print("1. Add new person")
            print("2. Start detection and recognition")
            print("3. Exit")

            try:
                choice = int(input("Enter your choice: "))
                if choice == 1:
                    self.add_person()
                elif choice == 2:
                    person_name = self.start_recognition()
                    print(person_name)
                elif choice == 3:
                    print("Exiting the system. Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")



if __name__ == "__main__":
    try:
        face_recognition = FaceRecognitionSystem()
        face_recognition.menu()
    except Exception as e:
        print(f"\nApplication error: {str(e)}")
