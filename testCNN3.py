import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)







def normalize_keypoints(keypoints):

    keypoints = np.array(keypoints).reshape(-1, 3)

    center = keypoints[0, :2]
    keypoints[:, :2] -= center
    max_distance = np.linalg.norm(keypoints[:, :2], axis=1).max()
    keypoints[:, :2] /= max_distance
    return keypoints.flatten()


def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        keypoints = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]
        )
        return keypoints
    else:
        return np.zeros((21, 3))




def augment_keypoints(sequence):

    angle = np.random.uniform(-20, 20)
    scale = np.random.uniform(0.9, 1.1)
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])
    augmented_sequence = []
    for frame in sequence:
        frame = frame.reshape(21, 3)
        frame[:, :2] = scale * np.dot(frame[:, :2], rotation_matrix.T)
        augmented_sequence.append(frame)
    return np.array(augmented_sequence)







def generate_label_map(data_path):

    labels = set()
    for outer_folder in os.listdir(data_path):
        outer_path = os.path.join(data_path, outer_folder)
        if not os.path.isdir(outer_path):
            continue

        for gesture_folder in os.listdir(outer_path):
            if '_' in gesture_folder:
                gesture_label = gesture_folder.split('_')[-1]
                labels.add(gesture_label)


    gesture_labels = sorted(list(labels))
    label_map = {gesture: idx for idx, gesture in enumerate(gesture_labels)}
    return gesture_labels, label_map


def load_data(data_path):
    sequences, labels = [], []
    for outer_folder in os.listdir(data_path):
        outer_path = os.path.join(data_path, outer_folder)
        if not os.path.isdir(outer_path):
            continue

        for gesture_folder in os.listdir(outer_path):
            gesture_path = os.path.join(outer_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            gesture_label = gesture_folder.split('_')[-1]
            if gesture_label not in label_map:
                continue

            for image_file in os.listdir(gesture_path):
                image_path = os.path.join(gesture_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                keypoints = extract_keypoints(image)
                if keypoints.size == 0:
                    print(f"Failed to extract keypoints: {image_path}")
                    continue
                if keypoints.shape != (21, 3):
                    print(f"Inconsistent shape: {keypoints.shape} in {image_path}")
                    continue
                sequences.append(keypoints.flatten())
                labels.append(label_map[gesture_label])

    if len(sequences) == 0 or len(labels) == 0:
        raise ValueError("No valid data found. Check the dataset structure and preprocessing functions.")

    return np.array(sequences), np.array(labels)



data_path = 'leapGestRecog'


gesture_labels, label_map = generate_label_map(data_path)

print("Gesture Labels:", gesture_labels)
print("Label Map:", label_map)


sequences, labels = load_data(data_path)


if sequences.size == 0:
    raise ValueError("Sequences array is empty. Check the data loading process.")
if np.max(sequences) == 0:
    raise ValueError("All sequence values are zero. Normalization cannot proceed.")


sequences = sequences / np.max(sequences)


sequences = sequences.reshape(-1, 21, 3, 1)


X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)


model = models.Sequential()


model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(21, 3, 1), padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))


model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(gesture_labels), activation='softmax'))


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)


model.save('gesture_recognition_model.keras')
