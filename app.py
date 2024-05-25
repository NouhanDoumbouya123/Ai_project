from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load your model
model_path = os.path.join(os.path.dirname(__file__), 'ViolenceDetectionModel.h5')
MoBiLSTM_model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Process the video file
        file_path = os.path.join(os.path.dirname(__file__), file.filename)
        file.save(file_path)
        
        # Here you would add your video processing and prediction logic
        cap = cv2.VideoCapture(file_path)
        predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Preprocess the frame and make predictions
            # Example preprocessing and prediction
            frame = cv2.resize(frame, (224, 224))
            frame = np.expand_dims(frame, axis=0)
            prediction = MoBiLSTM_model.predict(frame)
            predictions.append(prediction)
            
        cap.release()
        os.remove(file_path)
        
        return jsonify(predictions)
    
if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
