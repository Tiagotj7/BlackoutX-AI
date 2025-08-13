import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class NLPTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.encoder = LabelEncoder()
        self.model = None

    def load_data(self, filepath='nlp_model/intents.json'):
        with open(filepath) as file:
            data = json.load(file)
        
        texts = []
        labels = []
        
        for intent in data['intents']:
            for pattern in intent['patterns']:
                texts.append(pattern)
                labels.append(intent['tag'])
        
        return texts, labels

    def preprocess_data(self, texts, labels):
        # Vetorização dos textos
        X = self.vectorizer.fit_transform(texts).toarray()
        
        # Codificação das labels
        y = self.encoder.fit_transform(labels)
        
        return X, y

    def build_model(self, input_shape, num_classes):
        model = Sequential([
            Dense(128, input_shape=(input_shape,), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model

    def train(self, epochs=200):
        texts, labels = self.load_data()
        X, y = self.preprocess_data(texts, labels)
        
        self.model = self.build_model(X.shape[1], len(np.unique(y)))
        self.model.fit(X, y, epochs=epochs, batch_size=8, verbose=1)
        
        # Salvar modelo e processadores
        self.model.save('nlp_model/model.h5')
        with open('nlp_model/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('nlp_model/encoder.pkl', 'wb') as f:
            pickle.dump(self.encoder, f)

if __name__ == '__main__':
    trainer = NLPTrainer()
    trainer.train()