import json
import random
import tensorflow as tf
import numpy as np
from text_processor import TextProcessor
from conversation_memory import ConversationMemory
import uuid
from knowledge_base import init_cybersecurity_kb

class ResponseGenerator:
    def __init__(self):
        self.processor = TextProcessor()
        self.model = tf.keras.models.load_model('nlp_model/model.h5')
        self.intents = self.load_intents()
        self.memory = ConversationMemory(max_length=3)
        self.knowledge_base = init_cybersecurity_kb()
        
    def load_intents(self):
        with open('nlp_model/intents.json') as file:
            data = json.load(file)
        return data['intents']
    
    def predict_intent(self, text):
        # Usar contexto da conversa
        context = self.memory.get_context()
        full_text = f"{context} [SEP] {text}" if context else text
        
        vector = self.processor.vectorize_text(full_text)
        pred = self.model.predict(vector, verbose=0)
        intent_idx = np.argmax(pred)
        confidence = np.max(pred)
        return self.processor.encoder.classes_[intent_idx], confidence
    
    def get_response(self, intent_tag, user_input):
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                
                # Resposta criativa 30% das vezes
                if random.random() > 0.7:
                    return self.creative_response(response, user_input)
                return response
                
        return "⚠️ <strong>Erro no Sistema</strong><br><br>Comando não reconhecido. Diga 'ajuda' para opções."
    
    def creative_response(self, base_response, user_input):
        """Melhora respostas com variações criativas"""
        enhancements = [
            f"🔐 {base_response}",
            f"⚡ {base_response} <br><br>Dica: Use 'scan avançado' para mais detalhes.",
            f"🛡️ {base_response.replace('br>', 'br>➠ ')}"
        ]
        return random.choice(enhancements)

    def get_knowledge_based_response(self, user_input):
        results = self.knowledge_base.search(user_input, k=1)
        if not results:
            return None
            
        best_match = results[0]
        # Distância menor = mais similar (FAISS usa distância L2)
        if best_match['distance'] < 0.7:  # Limiar de similaridade
            return f"🔎 <strong>Base de Conhecimento</strong><br><br>{best_match['answer']}"
        return None

    def generate(self, user_input):
        # Tentar primeiro por intenção
        intent, confidence = self.predict_intent(user_input)
        
        # Se confiança baixa, buscar na base de conhecimento
        if confidence < 0.8:
            kb_response = self.get_knowledge_based_response(user_input)
            if kb_response:
                self.memory.add(user_input, kb_response)
                return kb_response, str(uuid.uuid4())
        
        response = self.get_response(intent, user_input)
        self.memory.add(user_input, response)
        return response, str(uuid.uuid4())