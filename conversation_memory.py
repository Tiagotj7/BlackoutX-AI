from collections import deque

class ConversationMemory:
    def __init__(self, max_length=5):
        self.memory = deque(maxlen=max_length)
        
    def add(self, user_input: str, ai_response: str):
        self.memory.append((user_input, ai_response))
        
    def get_context(self):
        return " | ".join([f"User: {inp} AI: {resp}" for inp, resp in self.memory])
    
    def clear(self):
        self.memory.clear()