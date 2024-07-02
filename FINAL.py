import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained DenseNet201 model
model = tf.keras.models.load_model('models/helmet_detection_model_densenet201.keras')

# Load pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_and_draw_boxes(frame, model):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_region = frame[y:y+h, x:x+w]
        
        # Resize face region to match model input size
        img = cv2.resize(face_region, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Perform prediction
        prediction = model.predict(img)
        
        # Interpret prediction
        label = np.argmax(prediction)
        confidence = np.max(prediction)

        if label == 0:  # Label 0 is "No Helmet"
            label_text = 'No Helmet'
            color = (0, 0, 255)  # Red
        else:  # Label 1 is "Helmet"
            label_text = 'Helmet'
            color = (0, 255, 0)  # Green
        
        # Draw bounding box and label on the original frame
        cv2.putText(frame, f'{label_text}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera is not available.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame, exiting loop.")
            break
        
        frame = predict_and_draw_boxes(frame, model)
        cv2.imshow('Helmet Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("An error occurred:", e)
    cap.release()
    cv2.destroyAllWindows()
