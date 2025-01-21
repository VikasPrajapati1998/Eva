import asyncio
from model import download_models
from face_recognize import FaceRecognitionSystem
from conversation import Conversation

async def main():
    # Download models if needed
    if not download_models():
        print("Error downloading models. Please try again.")
        return
    
    # Initialize and run face recognition system
    try:
        face_recognition = FaceRecognitionSystem()
        conv = Conversation()
        
        while True:
            print("Select an option: ")
            print("1. Add new person")
            print("2. Start detection and recognition")
            print("3. Exit")

            choice = int(input("Enter your choice: "))
            
            if choice == 1:
                face_recognition.add_person()
            elif choice == 2:
                person_name = face_recognition.start_recognition()
                await conv.talk(person_name)
            elif choice == 3:
                print("Exiting the system. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    except Exception as e:
        print(f"Error running face recognition system: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
