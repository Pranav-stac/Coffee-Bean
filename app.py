from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Load the trained YOLO model
model = YOLO('Models/best.pt')  # Path to your trained weights

def process_image(image_data):
    # Remove the base64 prefix
    image_data = image_data.split(',')[1]
    
    # Decode base64 to image
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(image)
    
    # Get predictions
    predictions = []
    for r in results[0].boxes.data:
        confidence = float(r[4])
        class_id = int(r[5])
        class_name = model.names[class_id]
        predictions.append({
            'class': class_name,
            'confidence': round(confidence * 100, 2)
        })
    
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        image_data = request.json['image']
        predictions = process_image(image_data)
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)