from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import psycopg2

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. DATABASE CONFIGURATION (For Supabase)
# ==========================================
DATABASE_URL = "postgresql://postgres:Manojkumar%40%40@db.sfrwkwfinyvymnqfiknl.supabase.co:5432/postgres"

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ==========================================
# 2. LIGHTWEIGHT NUMPY ML INFERENCE
# ==========================================
class NumpyModel:
    def __init__(self, weights_path):
        # Load the raw weights array exported from Colab
        with np.load(weights_path) as data:
            self.w1 = data['arr_0'] # Dense 1 Weights
            self.b1 = data['arr_1'] # Dense 1 Biases
            self.w2 = data['arr_2'] # Dense 2 Weights
            self.b2 = data['arr_3'] # Dense 2 Biases
        print("✅ Lightweight NumPy Model loaded successfully.")

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(self, x):
        # Layer 1: Dense (16, activation='relu')
        layer1_out = self.relu(np.dot(x, self.w1) + self.b1)
        # Layer 2: Dense (4, activation='softmax')
        layer2_out = self.softmax(np.dot(layer1_out, self.w2) + self.b2)
        return layer2_out

# Global model instance tracking
model = None
try:
    if os.path.exists('model_weights.npz'):
        model = NumpyModel('model_weights.npz')
    else:
        print("⚠️ Warning: model_weights.npz not found.")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize NumPy model. {e}")

PROFILES = ['aiml', 'bca', 'ca', 'it']

# ==========================================
# 3. ROUTES (ML Prediction)
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "ML Model weights are not loaded on the server."}), 500

    data = request.json
    user_answers = data.get('answers', [])
    
    if len(user_answers) != 8:
        return jsonify({"error": "Exactly 8 answers required"}), 400

    # Preprocess: Convert string answers back to our 32-length one-hot array
    x_encoded = np.zeros(32)
    counts = {'aiml': 0, 'bca': 0, 'ca': 0, 'it': 0}
    
    for q_idx, ans in enumerate(user_answers):
        counts[ans] += 1
        ans_idx = PROFILES.index(ans)
        x_encoded[q_idx * 4 + ans_idx] = 1

    # Predict using our lightweight NumPy matrix math
    input_tensor = np.array([x_encoded])
    predictions = model.predict(input_tensor)
    best_idx = np.argmax(predictions[0])
    predicted_profile = PROFILES[best_idx]

    # Calculate Radar Scores
    radar_data = [
      { "subject": 'Data Science', "A": round((counts['aiml'] / 8) * 100) },
      { "subject": 'Programming', "A": round((counts['bca'] / 8) * 100) },
      { "subject": 'User Experience', "A": round((counts['ca'] / 8) * 100) },
      { "subject": 'Infrastructure', "A": round((counts['it'] / 8) * 100) },
      { "subject": 'Logic / Math', "A": round(((counts['aiml'] + counts['bca']) / 16) * 100) }
    ]

    return jsonify({
        "profile": predicted_profile,
        "scores": radar_data
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
