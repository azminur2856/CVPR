import cv2
import numpy as np
import tensorflow as tf
import pickle

img_size = (160, 160)  # Define image size once

def real_time_recognition(model_path, label_dict_path, confidence_threshold=0.7, img_size=(160, 160)):
    model = tf.keras.models.load_model(model_path)
    with open(label_dict_path, 'rb') as f:
        label_dict = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, img_size)
            face_input = (face_resized.astype('float32') / 255.0).reshape(1, *img_size, 3)  # Dynamic shape

            preds = model.predict(face_input, verbose=0)[0]
            confidence = np.max(preds)
            pred_label = np.argmax(preds)

            label = f"{label_dict.get(pred_label, 'Unknown')} ({confidence:.2f})"
            color = (0, 255, 0) if confidence > confidence_threshold else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

real_time_recognition('face_recognition_model.keras', 'label_dict.pkl', img_size=img_size)