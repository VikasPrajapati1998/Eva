import os
import json

class QuestionHandler:
    def __init__(self, filename='unknown_questions.json'):
        self.filename = filename

    def save_question(self, question):
        # Check if the file exists
        if os.path.exists(self.filename):
            # Read the existing questions from the file
            with open(self.filename, 'r') as file:
                try:
                    questions = json.load(file)
                except json.JSONDecodeError:
                    questions = []  # In case the file is empty or has invalid JSON
        else:
            # If the file doesn't exist, initialize an empty list
            questions = []

        # Append the new question to the list
        questions.append({"question": question})

        # Write the updated list of questions back to the file
        with open(self.filename, 'w') as file:
            json.dump(questions, file, indent=4)

