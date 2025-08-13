import json
import random
import tensorflow as tf
import numpy as np
from text_processor import TextProcessor

class ResponseGenerator:
    def __init__(self):
        self.processor = TextProcessor()
        self.model = tf.keras.models.load_model('nlp_model/model.h5')
        self.intents = self.load_intents()
        
    def load_intents(self):
        with open('nlp_model/intents.json') as file:
            data = json.load(file)
        return data['intents']
    
    def predict_intent(self, text):
        vector = self.processor.vectorize_text(text)
        pred = self.model.predict(vector)
        intent_idx = np.argmax(pred)
        return self.processor.encoder.classes_[intent_idx]
    
    def get_response(self, intent_tag):
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "⚠️ <strong>Erro no Sistema</strong><br><br>Não entendi o comando. Repita ou diga 'ajuda'."

    def generate(self, user_input):
        intent = self.predict_intent(user_input)
        return self.get_response(intent)