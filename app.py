from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import psycopg2

app = Flask(__name__)
CORS(app) # Allows your React app to talk to this backend

# ==========================================
# 1. DATABASE CONFIGURATION (For Supabase)
# ==========================================
DATABASE_URL = "postgresql://postgres:Manojkumar%40%40@db.sfrwkwfinyvymnqfiknl.supabase.co:5432/postgres"

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ==========================================
# 2. MACHINE LEARNING CONFIGURATION
# ==========================================
try:
    # Changed extension from .h5 to .keras to match your new Colab download
    model = tf.keras.models.load_model('quiz_model.keras')
    print("✅ ML Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML model. {e}")
    model = None
    
PROFILES = ['aiml', 'bca', 'ca', 'it']

# ==========================================
# 3. ROUTES (ML Prediction)
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "ML Model is not loaded on the server."}), 500

    data = request.json
    user_answers = data.get('answers', []) # Expected: array of 8 profile strings
    
    if len(user_answers) != 8:
        return jsonify({"error": "Exactly 8 answers required"}), 400

    # Preprocess: Convert string answers back to our 32-length one-hot array
    x_encoded = np.zeros(32)
    counts = {'aiml': 0, 'bca': 0, 'ca': 0, 'it': 0}
    
    for q_idx, ans in enumerate(user_answers):
        counts[ans] += 1
        ans_idx = PROFILES.index(ans)
        x_encoded[q_idx * 4 + ans_idx] = 1

    # Predict using the model
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

    # Return to frontend
    return jsonify({
        "profile": predicted_profile,
        "scores": radar_data
    })

if __name__ == '__main__':
    # Using environment variable for port so it runs smoothly locally or on Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)