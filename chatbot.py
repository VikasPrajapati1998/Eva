import json
import re
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy
import nltk
from bs4 import BeautifulSoup
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
import numpy as np
from questionHandler import QuestionHandler


# Uncomment these lines if running for the first time
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

class ChatBot:
    def __init__(self, conversation_file):
        # Load conversation questions and answers from the JSON file
        with open(conversation_file, 'r') as file:
            self.conversations = json.load(file)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_md")  # Load spaCy model for semantic similarity
        
        # Initialize Feature Extractors
        self.vectorizer = TfidfVectorizer()
        self.count_vectorizer = CountVectorizer()
        
        # Prepare corpus for vectorizers and Word2Vec
        self.corpus = [self.preprocess_text(item['question']) for item in self.conversations]
        self.corpus = [' '.join(tokens) for tokens in self.corpus]
        
        # Fit the vectorizers
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.bow_matrix = self.count_vectorizer.fit_transform(self.corpus)
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec([tokens.split() for tokens in self.corpus], vector_size=100, window=5, min_count=1, workers=4)
        self.question_handler = QuestionHandler('unknown_questions.json')

    def preprocess_text(self, text):
        """
        Advanced Text Preprocessing:
        - Lowercase
        - Remove HTML tags
        - Remove URLs
        - Remove emojis
        - Remove special characters and punctuations
        - Remove stopwords
        - Chat word treatment
        - Spelling Correction
        - Lemmatization & Stemming
        """
        
        text = text.lower()     # Lowercase
        text = BeautifulSoup(text, "html.parser").get_text()    # Remove HTML tags
        text = re.sub(r'http\S+|www.\S+', '', text)     # Remove URLs
        text = emoji.replace_emoji(text, replace='')    # Remove emojis
        text = re.sub(r'[^\w\s]', '', text)     # Remove special characters and punctuations
        text = re.sub(r'\d+', '', text)     # Remove numbers
        tokens = text.split()   # Tokenization
        tokens = [word for word in tokens if word not in self.stop_words]   # Remove stopwords
        tokens = self.chat_word_treatment(tokens)   # Chat word treatment (expand contractions, handle slang, etc.)
        corrected_text = str(TextBlob(' '.join(tokens)).correct())  # Spelling Correction
        tokens = corrected_text.split()
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]     # Lemmatization
        tokens = [self.stemmer.stem(token) for token in tokens]     # Stemming
        
        return tokens

    def chat_word_treatment(self, tokens):
        """
        Handle chat-specific words like slang, abbreviations.
        """
        slang_dict = {
            "u": "you",
            "ur": "your",
            "idk": "I do not know",
            "im": "I am",
            "dont": "do not",
            "can't": "cannot",
            # Add more slang or abbreviations as needed
        }
        treated_tokens = [slang_dict.get(word, word) for word in tokens]
        return treated_tokens

    def calculate_semantic_similarity(self, user_input, question):
        """
        Calculate semantic similarity between user input and a stored question using spaCy.
        """
        user_doc = self.nlp(user_input)
        question_doc = self.nlp(question)
        return user_doc.similarity(question_doc)
    
    def calculate_word2vec_similarity(self, user_tokens, question_tokens):
        """
        Calculate similarity based on Word2Vec embeddings.
        """
        # Get vectors for user input
        user_vectors = []
        for token in user_tokens:
            if token in self.word2vec_model.wv:
                user_vectors.append(self.word2vec_model.wv[token])
        
        # Get vectors for question
        question_vectors = []
        for token in question_tokens:
            if token in self.word2vec_model.wv:
                question_vectors.append(self.word2vec_model.wv[token])
        
        if not user_vectors or not question_vectors:
            return 0.0
        
        user_avg = np.mean(user_vectors, axis=0)
        question_avg = np.mean(question_vectors, axis=0)
        
        # Cosine similarity
        cosine_sim = np.dot(user_avg, question_avg) / (np.linalg.norm(user_avg) * np.linalg.norm(question_avg))
        return cosine_sim
    
    def match_question(self, user_input):
        """
        Enhanced matching using multiple similarity measures:
        - Token Match Ratio
        - Semantic Similarity (spaCy)
        - Word2Vec Similarity
        """
        highest_match = 0
        best_response = "I didn't understand that. Could you try rephrasing?"
        
        user_tokens = self.preprocess_text(user_input)
        user_text = ' '.join(user_tokens)
        
        # # Transform user input using vectorizers
        # user_tfidf = self.vectorizer.transform([user_text])
        # user_bow = self.count_vectorizer.transform([user_text])
        
        for idx, item in enumerate(self.conversations):
            question = item['question']
            answer = item['answer']
            
            question_tokens = self.preprocess_text(question)
            question_text = ' '.join(question_tokens)
            
            # Similarity measures
            # Token Match Ratio & Semantic Similarity
            token_match_ratio = self.token_match_ratio(user_tokens, question_tokens)
            semantic_similarity = self.calculate_semantic_similarity(user_text, question_text)
            
            # Word2Vec Similarity
            word2vec_similarity = self.calculate_word2vec_similarity(user_tokens, question_tokens)
            
            # Combine similarities with weights
            combined_score = (0.4 * token_match_ratio) + (0.5 * semantic_similarity) + (0.4 * word2vec_similarity)
            
            if combined_score > 0.80 and combined_score > highest_match:
                highest_match = combined_score
                best_response = answer
            # Check for match between 50% and 80%
            elif 0.60 < combined_score <= 0.80 and combined_score > highest_match:
                highest_match = combined_score
                suggested_question = question
        
        # Return either the best response or a suggestion
        if highest_match > 0.80:
            return best_response
        elif 0.60 < highest_match <= 0.80 and suggested_question:
            self.question_handler.save_question(user_input)
            return f"Did you mean, {suggested_question} If yes please say this question."
        else:
            self.question_handler.save_question(user_input)
            return best_response
    

    def token_match_ratio(self, user_tokens, question_tokens):
        """
        Calculates a token-based matching ratio for two lists of tokens using SequenceMatcher.
        """
        user_text = ' '.join(user_tokens)
        question_text = ' '.join(question_tokens)
        match_ratio = SequenceMatcher(None, user_text, question_text).ratio()
        return match_ratio
    
    def respond(self, user_input):
        """
        Generate a response based on the matched question.
        """
        response = self.match_question(user_input)
        return response

# Example usage
if __name__ == "__main__":
    bot = ChatBot("conversation.json")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        response = bot.respond(user_input)
        print(f"Bot: {response}")





# pip install nltk spacy textblob beautifulsoup4 emoji scikit-learn gensim transformers
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_trf


