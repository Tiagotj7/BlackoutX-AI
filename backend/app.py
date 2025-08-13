from flask import Flask, request, jsonify
from response_generator import ResponseGenerator
import uuid

app = Flask(__name__)
ai_engine = ResponseGenerator()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # Gerar resposta da IA
    ai_response = ai_engine.generate(user_message)
    
    return jsonify({
        "response": ai_response,
        "message_id": str(uuid.uuid4())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)