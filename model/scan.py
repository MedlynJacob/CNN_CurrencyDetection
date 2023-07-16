import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('notes_cnn_model.h5')

# Define the currency class labels
class_labels = ['10 Rupees','20 Rupees','50 Rupees','100 Rupees','200 Rupees','500 Rupees']

# Set the desired video capture resolution
capture_width = 1280
capture_height = 720

# Start video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

while True:
    # Read each frame from the video capture
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)

    # Perform prediction
    predictions = model.predict(frame_input)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]

    # Draw rectangle around the detected currency

    # Display the predicted class label on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Currency Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
