from flask import Flask, request, jsonify
from response_generator import ResponseGenerator
import uuid
import os

app = Flask(__name__)
ai_engine = ResponseGenerator()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # Gerar resposta da IA
    ai_response, message_id = ai_engine.generate(user_message)
    
    return jsonify({
        "response": ai_response,
        "message_id": message_id
    })

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    message_id = data.get('message_id')
    correct_intent = data.get('correct_intent')
    
    if not message_id or not correct_intent:
        return jsonify({"error": "message_id and correct_intent are required"}), 400
    
    # Em produção: registrar feedback para retreino
    with open('feedback_logs.txt', 'a') as f:
        f.write(f"{message_id},{correct_intent}\n")
    
    return jsonify({"status": "Feedback received. Model will be updated."})

if __name__ == '__main__':
    os.makedirs('nlp_model', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)