import random

# Banco de sinônimos para segurança cibernética
SYNONYMS = {
    "analisar": ["verificar", "escanear", "investigar"],
    "dados": ["informações", "registros", "detalhes"],
    "remover": ["apagar", "deletar", "eliminar"],
    "digital": ["online", "eletrônica", "virtual"],
    "proteger": ["defender", "blindar", "fortalecer"],
    "senha": ["credencial", "acesso", "chave"]
}

def get_synonyms(word):
    return SYNONYMS.get(word.lower(), [word])

def augment_data(pattern: str):
    variations = []
    words = pattern.split()
    
    # Técnica 1: Substituição por sinônimos
    for _ in range(2):  # 2 variações por substituição
        new_words = []
        for word in words:
            if random.random() > 0.6:  # 40% de chance de substituir
                new_words.append(random.choice(get_synonyms(word)))
            else:
                new_words.append(word)
        variations.append(" ".join(new_words))
    
    # Técnica 2: Adição de palavras relevantes
    keywords = ["urgente", "prioridade", "confidencial", "criptografado"]
    for _ in range(2):
        new_pattern = pattern
        if random.random() > 0.5:
            new_pattern = f"{random.choice(keywords)} {new_pattern}"
        if random.random() > 0.5:
            new_pattern = f"{new_pattern} {random.choice(keywords)}"
        variations.append(new_pattern)
    
    # Técnica 3: Variações de ordem
    if len(words) > 3:
        shuffled = words.copy()
        random.shuffle(shuffled)
        variations.append(" ".join(shuffled))
    
    return variations