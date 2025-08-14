import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from data_augmentation import augment_data
from knowledge_base import KnowledgeBase

class NLPTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.encoder = LabelEncoder()
        self.model = None
        self.intents = None

    def load_intents(self, filepath='nlp_model/intents.json'):
        with open(filepath) as file:
            data = json.load(file)
        return data['intents']

    def load_data(self, filepath='nlp_model/intents.json'):
        self.intents = self.load_intents(filepath)
        
        texts = []
        labels = []
        
        for intent in self.intents:
            for pattern in intent['patterns']:
                texts.append(pattern)
                labels.append(intent['tag'])
                
                # Aumento de dados
                variations = augment_data(pattern)
                texts.extend(variations)
                labels.extend([intent['tag']] * len(variations))
        
        return texts, labels

    def preprocess_data(self, texts, labels):
        X = self.vectorizer.fit_transform(texts).toarray()
        y = self.encoder.fit_transform(labels)
        return X, y

    def build_model(self, input_shape, num_classes):
        model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=input_shape),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, epochs=200):
        texts, labels = self.load_data()
        X, y = self.preprocess_data(texts, labels)
        
        self.model = self.build_model(X.shape[1], len(np.unique(y)))
        self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=16, 
            validation_split=0.2,
            verbose=1
        )
        
        # Salvar artefatos
        self.model.save('nlp_model/model.h5')
        with open('nlp_model/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('nlp_model/encoder.pkl', 'wb') as f:
            pickle.dump(self.encoder, f)

    def create_knowledge_base(self):
        kb = KnowledgeBase()
        # Adicionar perguntas dos padrões como conhecimento
        for intent in self.intents:
            for pattern in intent['patterns']:
                # Usar a primeira resposta como resposta na base de conhecimento
                kb.add_entry(pattern, intent['responses'][0])
        
        # Adicionar conhecimento técnico adicional
        cybersecurity_faq = [
            ("Como criar uma senha forte?", "Use 12+ caracteres com letras, números e símbolos. Evite informações pessoais."),
            ("O que é phishing?", "Ataque onde criminosos se passam por entidades confiáveis para roubar dados sensíveis."),
            ("Como me proteger de ransomware?", "1. Backup regular 2. Atualizações de segurança 3. Não abrir anexos suspeitos"),
            ("O que é VPN?", "Rede Privada Virtual que criptografa sua conexão e mascara seu IP."),
            ("Como verificar vazamentos de dados?", "Use sites como HaveIBeenPwned ou Firefox Monitor."),
            ("O que é autenticação de dois fatores?", "Método de segurança que requer duas formas de verificação de identidade."),
            ("Como proteger meu roteador?", "1. Mude a senha padrão 2. Desative acesso remoto 3. Atualize firmware")
        ]
        
        for q, a in cybersecurity_faq:
            kb.add_entry(q, a)
        
        # Atualizar o índice FAISS e salvar
        kb.update_index()

if __name__ == '__main__':
    trainer = NLPTrainer()
    trainer.train()
    trainer.create_knowledge_base()
    print("Treinamento completo e base de conhecimento criada!")