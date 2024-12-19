import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Determine model path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'best.pt')

# Load the trained YOLO model
model = YOLO(MODEL_PATH)

def draw_bounding_boxes(image, results):
    """
    Draw bounding boxes on the image with labels and confidence
    
    Args:
    - image: Input image
    - results: YOLO detection results
    
    Returns:
    - Annotated image with bounding boxes
    """
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    
    for box in results[0].boxes.data:
        # Extract box coordinates and details
        x1, y1, x2, y2 = map(int, box[:4])
        confidence = float(box[4])
        class_id = int(box[5])
        class_name = model.names[class_id]
        
        # Define color (you can customize this)
        color = (0, 255, 0)  # Green color for bounding box
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label with class and confidence
        label = f"{class_name}: {confidence*100:.2f}%"
        cv2.putText(annotated_image, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return annotated_image

def process_image(image_data):
    """
    Process the input image, run detection, and generate predictions
    
    Args:
    - image_data: Base64 encoded image
    
    Returns:
    - Tuple of predictions and annotated image (base64 encoded)
    """
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
    
    # Draw bounding boxes
    annotated_image = draw_bounding_boxes(image, results)
    
    # Encode annotated image back to base64
    _, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return predictions, annotated_image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        image_data = request.json['image']
        predictions, annotated_image = process_image(image_data)
        return jsonify({
            'success': True, 
            'predictions': predictions,
            'annotated_image': f'data:image/jpeg;base64,{annotated_image}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, ssl_context=('cert.pem', 'key.pem'))
