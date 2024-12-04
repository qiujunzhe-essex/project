import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time


model = tf.keras.models.load_model('gesture_recognition_model.keras')


gesture_labels = ['c', '', 'fist','fistmoved','index', 'l', 'palmmoved', 'ok', 'palm', 'thumb','two','deaf']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


frame_sequence = deque(maxlen=30)


last_prediction_time = time.time()


prediction_interval = 0.5  # 秒


current_label = None


def extract_keypoints(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        keypoints = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]
        ).flatten()
    else:
        keypoints = np.zeros(21 * 3)
    return keypoints, results


def preprocess_frame(frame):
    keypoints, results = extract_keypoints(frame)
    frame_sequence.append(keypoints)


    if len(frame_sequence) == 30:

        latest_frame = np.array(frame_sequence)[-1].reshape(21, 3, 1)
        return latest_frame[np.newaxis, ...], results

    return None, results




cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    preprocessed_frame, results = preprocess_frame(frame)


    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )


    if preprocessed_frame is not None and time.time() - last_prediction_time >= prediction_interval:

        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = gesture_labels[predicted_class]


        current_label = predicted_label


        last_prediction_time = time.time()


    if current_label is not None:
        cv2.putText(frame, f'Predicted: {current_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Real-time Gesture Recognition", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
