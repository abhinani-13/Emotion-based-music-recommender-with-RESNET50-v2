from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from googleapiclient.discovery import build

app = Flask(__name__)

# YouTube API setup
YOUTUBE_API_KEY = 'AIzaSyB-xDNfVuMAvZ5ZZ5fxYfGO2AS4abvFrmQ'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Load models with error handling
cascade_path = '/Users/harshareddy/Documents/Major_project/Music_Recommendation_System_Through_Expression_Recognition-master/Main/haarcascade_frontalface_default (1).xml'
face_classifier = cv2.CascadeClassifier(cascade_path)
if face_classifier.empty():
    print(f"Error: Could not load Haar cascade file from {cascade_path}")
    exit(1)

classifier = load_model('/Users/harshareddy/Documents/Major_project/Music_Recommendation_System_Through_Expression_Recognition-master/Main/ResNet50V2_Model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
music_recommendations = {
    'Happy': 'happy days song',
    'Neutral': 'pushpa pushpa song',
    'Fear': 'devara song',
    'Sad': 'uniporaadhey song',
    'Surprise': 'salaar bgm',
    'Angry': 'rolex bgm',
    'No Faces': 'relaxing music'
}

def process_frame(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        label = "No Faces"
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

            if np.sum([roi_rgb]) != 0:
                roi = roi_rgb.astype('float32') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
        return label
    except Exception as e:
        print(f"Error processing frame: {e}")
        return "Error"

def get_youtube_video_id(query):
    try:
        request = youtube.search().list(
            part='id',
            q=query,
            type='video',
            maxResults=1
        )
        response = request.execute()
        if 'items' in response and len(response['items']) > 0:
            return response['items'][0]['id']['videoId']
        return None
    except Exception as e:
        print(f"Error fetching YouTube video ID: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')  # Corrected to use just 'index.html'

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        emotion = process_frame(frame)
        music = music_recommendations.get(emotion, 'relaxing music')
        video_id = get_youtube_video_id(music)
        return jsonify({'emotion': emotion, 'music': music, 'video_id': video_id})
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        return jsonify({'emotion': 'Error', 'music': 'Error', 'video_id': None}), 500

if __name__ == '__main__':
    app.run(debug=True)
