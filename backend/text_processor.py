import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:
    def __init__(self):
        self.vectorizer = None
        self.encoder = None
        self.load_artifacts()
        
    def load_artifacts(self):
        try:
            with open('nlp_model/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('nlp_model/encoder.pkl', 'rb') as f:
                self.encoder = pickle.load(f)
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            raise

    def preprocess_text(self, text):
        # Limpeza básica
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def vectorize_text(self, text):
        processed = self.preprocess_text(text)
        return self.vectorizer.transform([processed]).toarray()