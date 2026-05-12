import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import gemini_service
import utils
import tempfile

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "Gemini Vision + CNN-ViT Signal", "mode": "gemini"})

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filename = file.filename.lower()

        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_bytes = file.read()
            mime_type = "image/jpeg" if filename.endswith(('.jpg', '.jpeg')) else "image/png"
            result_class, result_confidence, result_score = gemini_service.analyze_content(image_bytes, mime_type)

        elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                result_class, result_confidence, result_score = gemini_service.analyze_content(tmp_path, "video/mp4")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        else:
            return jsonify({"error": "Unsupported file type"}), 400

        return jsonify({
            "class": result_class,
            "confidence": result_confidence,
            "confidence_score": result_score
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data.get('message')
    history = data.get('history', [])
    response_text = gemini_service.chat_response(history, user_message)

    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
