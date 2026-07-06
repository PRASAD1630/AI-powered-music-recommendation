import cv2
import numpy as np
import json
import time
import random
import webbrowser
from tensorflow.keras.models import load_model
import os

# ================================
# CONFIGURATION
# ================================
CAPTURE_DURATION = 60     # seconds to capture emotion
SONG_PLAY_DURATION =60       # seconds before capturing again
MODEL_PATH = 'emotion_model.h5'
CLASS_MAP_PATH = 'class_indices.json'

# ================================
# SONG RECOMMENDATIONS (YouTube + Spotify)
# ================================
recommendations = {
    'happy': [
        'https://open.spotify.com/track/7qiZfU4dY1lWllzX7mPBI3',  # Ed Sheeran – Shape of You
        'https://open.spotify.com/track/6fTt0CH2t0mdeB2NwCzFfP',
        'https://www.youtube.com/watch?v=kJQP7kiw5Fk',  # Despacito
        'https://www.youtube.com/watch?v=ZbZSe6N_BXs',  # Happy
        'https://open.spotify.com/track/2f8Z1P0JdE3U8j8p6d6X2h',  # Butta Bomma (Telugu)
        'https://www.youtube.com/watch?v=JtpN9TQHjjw'   # Ramulo Ramula
    ],
    'sad': [
        'https://open.spotify.com/track/5B3oE3Kj9WDAhX5Zs9gA7B',  # Tum Hi Ho
        'https://open.spotify.com/track/0XlEsQmmd0cZffr8D63XlH',  # Samajavaragamana
        'https://www.youtube.com/watch?v=9t3vYkY8f9Q',  # Sad Hindi song
        'https://www.youtube.com/watch?v=5vVb0k1U_1E'
    ],
    'angry': [
        'https://open.spotify.com/track/3AJwUDP919kvQ9QcozQPxg',
        'https://open.spotify.com/track/6EOKw2Z8d0wFpRmcE1mKUk',  # Nuvvosthanante (Energetic)
        'https://www.youtube.com/watch?v=hTWKbfoikeg',  # Nirvana – Smells Like Teen Spirit
        'https://www.youtube.com/watch?v=HCjNJDNzw8Y'
    ],
    'surprise': [
        'https://www.youtube.com/watch?v=SlPhMPnQ58k',  # Maroon 5
        'https://open.spotify.com/track/5af1515aRdR9a4sOfQj5f0',
        'https://www.youtube.com/watch?v=ZbZSe6N_BXs'
    ],
    'neutral': [
        'https://open.spotify.com/track/7BKLCZ1jbUBVqRi2FVlTVw',  # Closer
        'https://open.spotify.com/track/2Vv-BfVoq4g',  # Perfect
        'https://open.spotify.com/track/5vYA1mW9g2Coh1HUFusGZx',
        'https://www.youtube.com/watch?v=4NRXx6U8ABQ'
    ],
    'fear': [
        'https://www.youtube.com/watch?v=ZQ2nCGawrSY',
        'https://open.spotify.com/track/3KkXRkHbMCARz0aVfEt68P'
    ],
    'disgust': [
        'https://www.youtube.com/watch?v=VbfpW0pbvaU',
        'https://open.spotify.com/track/4xkOaSrkexMciUUogZKVTS'
    ]
}

# ================================
# LOAD MODEL + CLASS MAP
# ================================
def load_class_map(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if list(data.keys())[0].isalpha():
        data = {v: k for k, v in data.items()}
    return {int(k): v for k, v in data.items()}

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (48,48))
    face_resized = face_resized.astype('float32') / 255.0
    face_resized = np.expand_dims(face_resized, axis=-1)
    face_resized = np.expand_dims(face_resized, axis=0)
    return face_resized

# ================================
# MAIN LOOP
# ================================
def main():
    model = load_model(MODEL_PATH)
    class_map = load_class_map(CLASS_MAP_PATH)

    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    print("Starting Emotion Detection Cycle...")
    while True:
        print("\n🟢 Capturing emotion for 20 seconds...")
        start_time = time.time()
        emotion_counts = {}

        while (time.time() - start_time) < CAPTURE_DURATION:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                input_img = preprocess_face(roi)
                preds = model.predict(input_img, verbose=0)
                class_idx = int(np.argmax(preds))
                emotion = class_map[class_idx]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow("Emotion Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Pick dominant emotion
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            print(f"🎭 Dominant emotion: {dominant_emotion}")
        else:
            dominant_emotion = "neutral"
            print("No face detected, defaulting to neutral.")

        # Play recommendation
        links = recommendations.get(dominant_emotion.lower(), [])
        if links:
            chosen_link = random.choice(links)
            print(f"🎵 Opening song for '{dominant_emotion}': {chosen_link}")
            webbrowser.open_new_tab(chosen_link)
        else:
            print(f"No recommendation found for emotion '{dominant_emotion}'")

        print(f"🎧 Playing for {SONG_PLAY_DURATION} seconds...")
        time.sleep(SONG_PLAY_DURATION)

        print("🔁 Restarting emotion capture...\n")

if __name__ == '__main__':
    main()
