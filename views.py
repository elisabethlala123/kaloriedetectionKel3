from flask import render_template, request, jsonify, Response
from app import app
from app.models import model, device
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
import io
import base64
import numpy as np
import cv2
import torchvision.transforms as transforms

calorie_info = {
    'Tahu Goreng': 78,
    'Telur Rebus': 154,
    'Tempe Goreng': 35,
    'Nasi Putih': 129,
    'Telur Dadar': 93,
    'Tumis Kangkung': 98,
    'Paha Ayam Goreng': 207
}


# Daftar kelas makanan
category_names_with_bg = [
    'Background', 'Nasi Putih', 'Paha Ayam Goreng', 'Tahu Goreng',
    'Telur Dadar', 'Telur Rebus', 'Tempe Goreng', 'Tumis Kangkung'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    result_images = []
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    threshold = 0.9  # Set the threshold to 90%

    total_calories = 0
    detected_items = []

    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score >= threshold and label < len(category_names_with_bg):  # Ensure valid label and confidence above threshold
            class_name = category_names_with_bg[label]
            box = box.cpu().numpy().astype(int)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), class_name, fill="red", font=font)
            
            cropped_image = image.crop((box[0], box[1], box[2], box[3]))
            buffered = io.BytesIO()
            cropped_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            result_images.append({"class_name": class_name, "img_str": img_str})

            # Calculate total calories and keep track of detected items
            if class_name in calorie_info:
                item_calories = calorie_info[class_name]
                total_calories += item_calories
                detected_items.append({"class_name": class_name, "calories": item_calories})

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    annotated_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Calculate percentage of total calories for each item
    for item in detected_items:
        item['percentage'] = (item['calories'] / total_calories) * 100

    response = {
        'annotated_img_str': annotated_img_str,
        'results': result_images,
        'total_calories': total_calories,
        'detected_items': detected_items
    }

    return jsonify(response)

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/realtime_detection')
def realtime_detection():
    return render_template('realtime_detection.html')

def detect():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        image_tensor = transform(image).to(device).unsqueeze(0)

        with torch.no_grad():
            prediction = model(image_tensor)[0]

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        threshold = 0.9

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score >= threshold and label < len(category_names_with_bg):
                class_name = category_names_with_bg[label]
                box = box.cpu().numpy().astype(int)
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
                draw.text((box[0], box[1]), class_name, fill="red", font=font)

        # Convert PIL image back to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

