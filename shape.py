import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 
# Function to detect and analyze shapes
def check_shapes_inside(image_path, fill_closed=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to load image at {image_path}")
    
    edges = cv2.Canny(img, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_with_markers = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    closed_shapes = 0
    open_shapes = 0

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            continue

        start_point = contour[0][0]
        end_point = contour[-1][0]
        distance = np.linalg.norm(start_point - end_point)
        tolerance = 5

        if distance <= tolerance:
            closed_shapes += 1
            if fill_closed:
                cv2.drawContours(img_with_markers, [contour], -1, (0, 255, 0), -1)
        else:
            open_shapes += 1
            cv2.drawContours(img_with_markers, [contour], -1, (0, 0, 255), 2)
            cv2.circle(img_with_markers, tuple(start_point), 5, (255, 0, 0), -1)
            cv2.circle(img_with_markers, tuple(end_point), 5, (0, 255, 0), -1)

    _, buffer = cv2.imencode('.png', img_with_markers)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return closed_shapes, open_shapes, img_str


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})
    
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    closed_shapes, open_shapes, img_str = check_shapes_inside(filepath, fill_closed=False)
    os.remove(filepath)

    return jsonify({
        'closed_shapes': closed_shapes,
        'open_shapes': open_shapes,
        'img_str': img_str
    })


@app.route('/fill_closed_shapes', methods=['POST'])
def fill_closed_shapes():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    closed_shapes, open_shapes, img_str = check_shapes_inside(filepath, fill_closed=True)
    os.remove(filepath)

    return jsonify({
        'closed_shapes': closed_shapes,
        'open_shapes': open_shapes,
        'img_str': img_str
    })
@app.route('/generate_points', methods=['POST'])
def generate_points():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shapes_points = [contour.squeeze().tolist() for contour in contours]
    os.remove(filepath)

    return jsonify({'shapes_points': shapes_points})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
