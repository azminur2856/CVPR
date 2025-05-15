from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load model and labels
model = tf.keras.models.load_model('face_recognition_model.keras')
with open('label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    return (face_resized.astype('float32') / 255.0).reshape(1, 160, 160, 3)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_input = preprocess_face(face)
            preds = model.predict(face_input, verbose=0)[0]
            confidence = np.max(preds)
            label = label_dict[np.argmax(preds)] if confidence > 0.7 else "Unknown"
            
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)