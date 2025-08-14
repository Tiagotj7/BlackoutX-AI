import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import os

class KnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.index = None
        self.knowledge_df = pd.DataFrame(columns=['question', 'answer'])
        self.index_path = 'nlp_model/knowledge_base.faiss'
        self.csv_path = 'nlp_model/knowledge_base.csv'
        
        if os.path.exists(self.index_path) and os.path.exists(self.csv_path):
            self.load_index()
        else:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
    def add_entry(self, question: str, answer: str):
        new_entry = pd.DataFrame([[question, answer]], columns=['question', 'answer'])
        self.knowledge_df = pd.concat([self.knowledge_df, new_entry], ignore_index=True)
    
    def update_index(self):
        if self.knowledge_df.empty:
            print("Aviso: Base de conhecimento vazia. Índice não criado.")
            return
            
        questions = self.knowledge_df['question'].tolist()
        embeddings = self.model.encode(questions)
        
        # Normalizar vetores para uso com IndexFlatIP (produto interno)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
        
        # Criar índice FAISS (usando produto interno para similaridade cosseno)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        
        # Salvar
        faiss.write_index(self.index, self.index_path)
        self.knowledge_df.to_csv(self.csv_path, index=False)
        print(f"Índice FAISS atualizado com {len(self.knowledge_df)} entradas.")
    
    def load_index(self):
        try:
            self.index = faiss.read_index(self.index_path)
            self.knowledge_df = pd.read_csv(self.csv_path)
            print(f"Base de conhecimento carregada com {len(self.knowledge_df)} entradas.")
        except Exception as e:
            print(f"Erro ao carregar base de conhecimento: {e}")
            self.index = None
            self.knowledge_df = pd.DataFrame(columns=['question', 'answer'])
    
    def search(self, query: str, k=3):
        if self.index is None or self.knowledge_df.empty:
            return []
            
        query_embed = self.model.encode([query])
        query_embed = query_embed / np.linalg.norm(query_embed)
        distances, indices = self.index.search(query_embed.astype(np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # -1 indica nenhum resultado
                results.append({
                    'question': self.knowledge_df.iloc[idx]['question'],
                    'answer': self.knowledge_df.iloc[idx]['answer'],
                    'distance': distances[0][i]  # Similaridade cosseno (quanto maior, mais similar)
                })
        # Ordenar por distância (maior similaridade primeiro)
        results.sort(key=lambda x: x['distance'], reverse=True)
        return results

# Função para inicializar com conhecimento básico
def init_cybersecurity_kb():
    kb = KnowledgeBase()
    # Se a base estiver vazia, adicionar conhecimento mínimo
    if kb.knowledge_df.empty:
        kb.add_entry("Como criar uma senha forte?", "Use 12+ caracteres com letras, números e símbolos. Evite informações pessoais.")
        kb.add_entry("O que é phishing?", "Ataque onde criminosos se passam por entidades confiáveis para roubar dados sensíveis.")
        kb.add_entry("Como me proteger de ransomware?", "1. Backup regular 2. Atualizações de segurança 3. Não abrir anexos suspeitos")
        kb.update_index()
    return kb