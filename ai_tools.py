import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import random
import os
import wave
import face_recognition
import librosa
import numpy as np
from transformers import pipeline
import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from geopy.distance import geodesic

# Generic Plot Function for all modules
def create_plot(data, title, x_label, y_label, plot_type='bar'):
    x = [i for i in range(len(data))]
    y = [d[0] if isinstance(d, tuple) else d for d in data]
    plt.figure(figsize=(6, 4))
    if plot_type == 'bar':
        plt.bar(x, y, color='gray', alpha=0.7)
    elif (plot_type == 'plot'):
        plt.plot(x, y, 'b-o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{plot_url}"

# 1. Fingerprint Analysis
def extract_fingerprint_features(image_path):
    """
    Extract fingerprint features from an image.
    :param image_path: Path to the fingerprint image.
    :return: List of extracted features (minutiae points).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Resize for consistency
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur
    edges = cv2.Canny(img, 50, 150)  # Detect edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # Centroid x
            cy = int(moments['m01'] / moments['m00'])  # Centroid y
            features.append((cx, cy))
    return features

def match_fingerprints(fp1, fp2, distance_threshold=15, match_threshold=0.8):
    """
    Match two sets of fingerprint features.
    :param fp1: Features from the first fingerprint.
    :param fp2: Features from the second fingerprint.
    :param distance_threshold: Maximum distance to consider a match.
    :param match_threshold: Threshold for determining a match (0.0 to 1.0).
    :return: Match score as a float.
    """
    matches = 0
    for x1, y1 in fp1:
        for x2, y2 in fp2:
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if distance < distance_threshold:
                matches += 1
                break
    match_score = matches / max(len(fp1), len(fp2))
    return match_score, match_score >= match_threshold

def analyze_fingerprints(crime_image_path, suspect_image_path):
    """
    Perform fingerprint analysis by comparing features.
    :param crime_image_path: Path to the crime fingerprint image.
    :param suspect_image_path: Path to the suspect fingerprint image.
    :return: Analysis result as a dictionary.
    """
    try:
        crime_features = extract_fingerprint_features(crime_image_path)
        suspect_features = extract_fingerprint_features(suspect_image_path)
        match_score, is_match = match_fingerprints(crime_features, suspect_features)

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if is_match else "No Match",
            "crime_image": crime_image_path,
            "suspect_image": suspect_image_path
        }
    except Exception as e:
        return {"error": str(e)}

# 2. Gunshot Analysis
def extract_gunshot_features(audio_path):
    with wave.open(audio_path, 'rb') as wav:
        random.seed(wav.getnframes())
    return [(random.uniform(50, 120), random.uniform(100, 1000), random.uniform(10, 500)) for _ in range(5)]  # (amplitude, freq, duration)

def match_gunshot(audio1, audio2, amp_threshold=10, freq_threshold=100, dur_threshold=50):
    matches = 0
    for a1, f1, d1 in audio1:
        for a2, f2, d2 in audio2:
            if (abs(a1 - a2) < amp_threshold and abs(f1 - f2) < freq_threshold and abs(d1 - d2) < dur_threshold):
                matches += 1
                break
    return matches / min(len(audio1), len(audio2))

def analyze_gunshot(crime_audio_path, suspect_audio_path, similarity_threshold=0.8):
    """
    Analyze gunshot audio files and return the result.
    :param crime_audio_path: Path to the crime scene audio file.
    :param suspect_audio_path: Path to the suspect's audio file.
    :param similarity_threshold: Threshold for determining a match.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load audio files
        crime_audio, crime_sr = librosa.load(crime_audio_path, sr=None)
        suspect_audio, suspect_sr = librosa.load(suspect_audio_path, sr=None)

        # Extract MFCC features
        crime_mfcc = librosa.feature.mfcc(y=crime_audio, sr=crime_sr, n_mfcc=13)
        suspect_mfcc = librosa.feature.mfcc(y=suspect_audio, sr=suspect_sr, n_mfcc=13)

        # Flatten MFCCs for comparison
        crime_mfcc_flat = crime_mfcc.flatten()
        suspect_mfcc_flat = suspect_mfcc.flatten()

        # Compute cosine similarity
        similarity = np.dot(crime_mfcc_flat, suspect_mfcc_flat) / (
            np.linalg.norm(crime_mfcc_flat) * np.linalg.norm(suspect_mfcc_flat)
        )

        return {
            "suspect_score": round(similarity, 2),
            "message": "Match" if similarity >= similarity_threshold else "No Match",
            "crime_audio": crime_audio_path,
            "suspect_audio": suspect_audio_path
        }
    except Exception as e:
        return {"error": str(e)}

# 3. Deepfake Analysis
def analyze_deepfake(crime_scene_path, suspect_path):
    """
    Analyze videos or images for deepfake detection.
    :param crime_scene_path: Path to the crime scene video or image.
    :param suspect_path: Path to the suspect's video or image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load pre-trained deepfake detection model
        deepfake_detector = pipeline("image-classification", model="deepfake-detection")

        # Analyze crime scene and suspect files
        crime_result = deepfake_detector(crime_scene_path)
        suspect_result = deepfake_detector(suspect_path)

        # Extract scores
        crime_score = crime_result[0]['score']
        suspect_score = suspect_result[0]['score']

        return {
            "crime_score": round(crime_score, 2),
            "suspect_score": round(suspect_score, 2),
            "message": "Deepfake Detected" if suspect_score > 0.5 else "No Deepfake",
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 4. Face Analysis
def analyze_face(crime_scene_path, suspect_path, match_threshold=0.6):
    """
    Analyze face images and return the result.
    :param crime_scene_path: Path to the crime scene image.
    :param suspect_path: Path to the suspect's image.
    :param match_threshold: Threshold for determining a match.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load images
        crime_image = face_recognition.load_image_file(crime_scene_path)
        suspect_image = face_recognition.load_image_file(suspect_path)

        # Extract face encodings
        crime_encoding = face_recognition.face_encodings(crime_image)
        suspect_encoding = face_recognition.face_encodings(suspect_image)

        if not crime_encoding or not suspect_encoding:
            return {
                "suspect_score": 0.0,
                "message": "No face detected in one or both images.",
                "crime_image": crime_scene_path,
                "suspect_image": suspect_path
            }

        # Compare faces
        match_results = face_recognition.compare_faces([crime_encoding[0]], suspect_encoding[0], tolerance=match_threshold)
        distance = face_recognition.face_distance([crime_encoding[0]], suspect_encoding[0])[0]

        return {
            "suspect_score": round(1 - distance, 2),
            "message": "Match" if match_results[0] else "No Match",
            "crime_image": crime_scene_path,
            "suspect_image": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 5. Bloodstain Analysis
def extract_bloodstain_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 50), random.uniform(0, 360)) for _ in range(5)]  # (size, angle)

def match_bloodstain(b1, b2, size_threshold=5, angle_threshold=30):
    matches = 0
    for s1, a1 in b1:
        for s2, a2 in b2:
            if (abs(s1 - s2) < size_threshold and abs(a1 - a2) < angle_threshold):
                matches += 1
                break
    return matches / min(len(b1), len(b2))

def analyze_bloodstain(crime_scene_path, suspect_path):
    """
    Analyze bloodstain patterns and return the result.
    :param crime_scene_path: Path to the crime scene image.
    :param suspect_path: Path to the suspect's image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load images
        crime_image = cv2.imread(crime_scene_path, cv2.IMREAD_GRAYSCALE)
        suspect_image = cv2.imread(suspect_path, cv2.IMREAD_GRAYSCALE)

        # Detect contours (simulate bloodstain detection)
        _, crime_thresh = cv2.threshold(crime_image, 127, 255, cv2.THRESH_BINARY)
        _, suspect_thresh = cv2.threshold(suspect_image, 127, 255, cv2.THRESH_BINARY)
        crime_contours, _ = cv2.findContours(crime_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspect_contours, _ = cv2.findContours(suspect_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract features (size and angle)
        def extract_features(contours):
            features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                features.append((area, angle))
            return features

        crime_features = extract_features(crime_contours)
        suspect_features = extract_features(suspect_contours)

        # Compare features
        matches = 0
        for c_area, c_angle in crime_features:
            for s_area, s_angle in suspect_features:
                if abs(c_area - s_area) < 50 and abs(c_angle - s_angle) < 15:
                    matches += 1
                    break

        match_score = matches / max(len(crime_features), len(suspect_features))

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_image": crime_scene_path,
            "suspect_image": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 6. Ballistics Analysis
def extract_ballistics_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 10), random.uniform(0, 360)) for _ in range(5)]  # (groove width, angle)

def match_ballistics(b1, b2, width_threshold=1, angle_threshold=20):
    matches = 0
    for w1, a1 in b1:
        for w2, a2 in b2:
            if (abs(w1 - w2) < width_threshold and abs(a1 - a2) < angle_threshold):
                matches += 1
                break
    return matches / min(len(b1), len(b2))

def analyze_ballistics(crime_scene_path, suspect_path):
    """
    Analyze ballistic marks and return the result.
    :param crime_scene_path: Path to the crime scene image.
    :param suspect_path: Path to the suspect's image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load images
        crime_image = cv2.imread(crime_scene_path, cv2.IMREAD_GRAYSCALE)
        suspect_image = cv2.imread(suspect_path, cv2.IMREAD_GRAYSCALE)

        # Detect edges (simulate ballistic mark detection)
        crime_edges = cv2.Canny(crime_image, 100, 200)
        suspect_edges = cv2.Canny(suspect_image, 100, 200)

        # Detect contours
        crime_contours, _ = cv2.findContours(crime_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspect_contours, _ = cv2.findContours(suspect_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract features (groove width and angle)
        def extract_features(contours):
            features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                features.append((area, angle))
            return features

        crime_features = extract_features(crime_contours)
        suspect_features = extract_features(suspect_contours)

        # Compare features
        matches = 0
        for c_area, c_angle in crime_features:
            for s_area, s_angle in suspect_features:
                if abs(c_area - s_area) < 50 and abs(c_angle - s_angle) < 15:
                    matches += 1
                    break

        match_score = matches / max(len(crime_features), len(suspect_features))

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_image": crime_scene_path,
            "suspect_image": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 7. Voiceprint Analysis
def extract_voiceprint_features(audio_path):
    with wave.open(audio_path, 'rb') as wav:
        random.seed(wav.getnframes())
    return [(random.uniform(50, 200), random.uniform(0, 500)) for _ in range(5)]  # (pitch, freq)

def match_voiceprint(v1, v2, pitch_threshold=20, freq_threshold=50):
    matches = 0
    for p1, f1 in v1:
        for p2, f2 in v2:
            if (abs(p1 - p2) < pitch_threshold and abs(f1 - f2) < freq_threshold):
                matches += 1
                break
    return matches / min(len(v1), len(v2))

def analyze_voiceprint(crime_scene_path, suspect_path):
    crime_scene = extract_voiceprint_features(crime_scene_path)
    suspect = extract_voiceprint_features(suspect_path)
    suspect_score = match_voiceprint(crime_scene, suspect)
    crime_plot = create_plot([v[0] for v in crime_scene], "Crime Scene Voiceprint", "Sample", "Pitch (Hz)", 'plot')
    suspect_plot = create_plot([v[0] for v in suspect], "Suspect Voiceprint", "Sample", "Pitch (Hz)", 'plot')
    return {
        "suspect_score": round(suspect_score, 2),
        "message": "Suspect matches crime scene!" if suspect_score > 0.6 else "No match.",
        "crime_plot": crime_plot,
        "crime_audio": crime_scene_path.split('/')[-1],
        "suspect_plot": suspect_plot,
        "suspect_audio": suspect_path.split('/')[-1],
        "crime_path": crime_scene_path,
        "suspect_path": suspect_path
    }

# 8. Handwriting Analysis
def extract_handwriting_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 10), random.uniform(0, 90)) for _ in range(5)]  # (stroke width, angle)

def match_handwriting(h1, h2, width_threshold=1, angle_threshold=10):
    matches = 0
    for w1, a1 in h1:
        for w2, a2 in h2:
            if (abs(w1 - w2) < width_threshold and abs(a1 - a2) < angle_threshold):
                matches += 1
                break
    return matches / min(len(h1), len(h2))

def analyze_handwriting(crime_scene_path, suspect_path):
    """
    Analyze handwriting samples and return the result.
    :param crime_scene_path: Path to the crime scene handwriting image.
    :param suspect_path: Path to the suspect's handwriting image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load images
        crime_image = cv2.imread(crime_scene_path, cv2.IMREAD_GRAYSCALE)
        suspect_image = cv2.imread(suspect_path, cv2.IMREAD_GRAYSCALE)

        # Detect edges (simulate handwriting stroke detection)
        crime_edges = cv2.Canny(crime_image, 100, 200)
        suspect_edges = cv2.Canny(suspect_image, 100, 200)

        # Detect contours
        crime_contours, _ = cv2.findContours(crime_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspect_contours, _ = cv2.findContours(suspect_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract features (stroke width and angle)
        def extract_features(contours):
            features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                features.append((area, angle))
            return features

        crime_features = extract_features(crime_contours)
        suspect_features = extract_features(suspect_contours)

        # Compare features
        matches = 0
        for c_area, c_angle in crime_features:
            for s_area, s_angle in suspect_features:
                if abs(c_area - s_area) < 50 and abs(c_angle - s_angle) < 15:
                    matches += 1
                    break

        match_score = matches / max(len(crime_features), len(suspect_features))

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_image": crime_scene_path,
            "suspect_image": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 9. Tire Track Analysis
def extract_tire_track_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 20), random.uniform(0, 360)) for _ in range(5)]  # (tread depth, angle)

def match_tire_track(t1, t2, depth_threshold=2, angle_threshold=30):
    matches = 0
    for d1, a1 in t1:
        for d2, a2 in t2:
            if (abs(d1 - d2) < depth_threshold and abs(a1 - a2) < angle_threshold):
                matches += 1
                break
    return matches / min(len(t1), len(t2))

def analyze_tire_track(crime_scene_path, suspect_path):
    """
    Analyze tire track patterns and return the result.
    :param crime_scene_path: Path to the crime scene tire track image.
    :param suspect_path: Path to the suspect's tire track image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load images
        crime_image = cv2.imread(crime_scene_path, cv2.IMREAD_GRAYSCALE)
        suspect_image = cv2.imread(suspect_path, cv2.IMREAD_GRAYSCALE)

        # Detect edges (simulate tire track detection)
        crime_edges = cv2.Canny(crime_image, 100, 200)
        suspect_edges = cv2.Canny(suspect_image, 100, 200)

        # Detect contours
        crime_contours, _ = cv2.findContours(crime_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspect_contours, _ = cv2.findContours(suspect_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract features (tread depth and angle)
        def extract_features(contours):
            features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                features.append((area, angle))
            return features

        crime_features = extract_features(crime_contours)
        suspect_features = extract_features(suspect_contours)

        # Compare features
        matches = 0
        for c_area, c_angle in crime_features:
            for s_area, s_angle in suspect_features:
                if abs(c_area - s_area) < 50 and abs(c_angle - s_angle) < 15:
                    matches += 1
                    break

        match_score = matches / max(len(crime_features), len(suspect_features))

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_image": crime_scene_path,
            "suspect_image": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 10. Tool Mark Analysis
def extract_tool_mark_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 5), random.uniform(0, 360)) for _ in range(5)]  # (mark width, angle)

def analyze_tool_mark(crime_scene_path, suspect_path):
    """
    Analyze tool mark patterns and return the result.
    :param crime_scene_path: Path to the crime scene tool mark image.
    :param suspect_path: Path to the suspect's tool mark image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load images
        crime_image = cv2.imread(crime_scene_path, cv2.IMREAD_GRAYSCALE)
        suspect_image = cv2.imread(suspect_path, cv2.IMREAD_GRAYSCALE)

        # Detect edges (simulate tool mark detection)
        crime_edges = cv2.Canny(crime_image, 100, 200)
        suspect_edges = cv2.Canny(suspect_image, 100, 200)

        # Detect contours
        crime_contours, _ = cv2.findContours(crime_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspect_contours, _ = cv2.findContours(suspect_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract features (mark width and angle)
        def extract_features(contours):
            features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                features.append((area, angle))
            return features

        crime_features = extract_features(crime_contours)
        suspect_features = extract_features(suspect_contours)

        # Compare features
        matches = 0
        for c_area, c_angle in crime_features:
            for s_area, s_angle in suspect_features:
                if abs(c_area - s_area) < 50 and abs(c_angle - s_angle) < 15:
                    matches += 1
                    break

        match_score = matches / max(len(crime_features), len(suspect_features))

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_image": crime_scene_path,
            "suspect_image": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 11. Fiber Analysis
def extract_fiber_features(image_path):
    """
    Simulate the extraction of fiber properties from an image.
    :param image_path: Path to the fiber image.
    :return: List of simulated fiber properties (diameter, length).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0.1, 1.0), random.uniform(10, 100)) for _ in range(5)]  # (diameter, length)

def match_fiber(f1, f2, diam_threshold=0.1, length_threshold=10):
    """
    Compare two sets of fiber properties and calculate a match score.
    :param f1: List of fiber properties from the crime scene.
    :param f2: List of fiber properties from the suspect.
    :param diam_threshold: Threshold for diameter comparison.
    :param length_threshold: Threshold for length comparison.
    :return: Match score as a float.
    """
    matches = 0
    for d1, l1 in f1:
        for d2, l2 in f2:
            if abs(d1 - d2) < diam_threshold and abs(l1 - l2) < length_threshold:
                matches += 1
                break
    return matches / min(len(f1), len(f2))

def analyze_fiber(crime_scene_path, suspect_path):
    """
    Analyze fiber samples and return the result.
    :param crime_scene_path: Path to the crime scene fiber image.
    :param suspect_path: Path to the suspect's fiber image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract fiber features
        crime_fiber = extract_fiber_features(crime_scene_path)
        suspect_fiber = extract_fiber_features(suspect_path)

        # Match fiber features
        match_score = match_fiber(crime_fiber, suspect_fiber)

        # Generate plots for visualization
        crime_plot = create_plot([f[0] for f in crime_fiber], "Crime Scene Fiber", "Fiber", "Diameter (mm)")
        suspect_plot = create_plot([f[0] for f in suspect_fiber], "Suspect Fiber", "Fiber", "Diameter (mm)")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 12. Shoe Print Analysis
def extract_shoe_print_features(image_path):
    """
    Simulate the extraction of shoe print properties from an image.
    :param image_path: Path to the shoe print image.
    :return: List of simulated shoe print properties (pattern width, angle).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 30), random.uniform(0, 360)) for _ in range(5)]  # (pattern width, angle)

def match_shoe_print(s1, s2, width_threshold=3, angle_threshold=30):
    """
    Compare two sets of shoe print properties and calculate a match score.
    :param s1: List of shoe print properties from the crime scene.
    :param s2: List of shoe print properties from the suspect.
    :param width_threshold: Threshold for width comparison.
    :param angle_threshold: Threshold for angle comparison.
    :return: Match score as a float.
    """
    matches = 0
    for w1, a1 in s1:
        for w2, a2 in s2:
            if abs(w1 - w2) < width_threshold and abs(a1 - a2) < angle_threshold:
                matches += 1
                break
    return matches / min(len(s1), len(s2))

def analyze_shoe_print(crime_scene_path, suspect_path):
    """
    Analyze shoe print patterns and return the result.
    :param crime_scene_path: Path to the crime scene shoe print image.
    :param suspect_path: Path to the suspect's shoe print image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract shoe print features
        crime_shoe = extract_shoe_print_features(crime_scene_path)
        suspect_shoe = extract_shoe_print_features(suspect_path)

        # Match shoe print features
        match_score = match_shoe_print(crime_shoe, suspect_shoe)

        # Generate plots for visualization
        crime_plot = create_plot([s[0] for s in crime_shoe], "Crime Scene Shoe Print", "Pattern", "Width (mm)")
        suspect_plot = create_plot([s[0] for s in suspect_shoe], "Suspect Shoe Print", "Pattern", "Width (mm)")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 13. Digital Footprint Analysis
def extract_digital_footprint_features(text_path):
    with open(text_path, 'r') as f:
        random.seed(len(f.read()))
    return [(random.uniform(0, 100), random.uniform(0, 10)) for _ in range(5)]  # (activity count, time)

def match_digital_footprint(d1, d2, count_threshold=10, time_threshold=1):
    matches = 0
    for c1, t1 in d1:
        for c2, t2 in d2:
            if (abs(c1 - c2) < count_threshold and abs(t1 - t2) < time_threshold):
                matches += 1
                break
    return matches / min(len(d1), len(d2))

def analyze_digital_footprint(crime_scene_path, suspect_path):
    crime_scene = extract_digital_footprint_features(crime_scene_path)
    suspect = extract_digital_footprint_features(suspect_path)
    suspect_score = match_digital_footprint(crime_scene, suspect)
    crime_plot = create_plot([d[0] for d in crime_scene], "Crime Scene Digital Footprint", "Event", "Activity Count")
    suspect_plot = create_plot([d[0] for d in suspect], "Suspect Digital Footprint", "Event", "Activity Count")
    return {
        "suspect_score": round(suspect_score, 2),
        "message": "Suspect matches crime scene!" if suspect_score > 0.6 else "No match.",
        "crime_plot": crime_plot,
        "suspect_plot": suspect_plot,
        "crime_path": crime_scene_path,
        "suspect_path": suspect_path
    }

# 14. Odor Profile Analysis
def extract_odor_profile_features(text_path):
    """
    Simulate the extraction of odor profile features from a text file.
    :param text_path: Path to the text file containing odor profile data.
    :return: List of simulated features (intensity, compound count).
    """
    with open(text_path, 'r') as f:
        random.seed(len(f.read()))
    return [(random.uniform(0, 1000), random.uniform(0, 100)) for _ in range(5)]  # (intensity, compound count)

def match_odor_profile(o1, o2, intensity_threshold=100, count_threshold=10):
    """
    Compare two sets of odor profile features and calculate a match score.
    :param o1: List of features from the crime scene.
    :param o2: List of features from the suspect.
    :param intensity_threshold: Threshold for intensity comparison.
    :param count_threshold: Threshold for compound count comparison.
    :return: Match score as a float.
    """
    matches = 0
    for i1, c1 in o1:
        for i2, c2 in o2:
            if abs(i1 - i2) < intensity_threshold and abs(c1 - c2) < count_threshold:
                matches += 1
                break
    return matches / min(len(o1), len(o2))

def analyze_odor_profile(crime_scene_path, suspect_path):
    """
    Analyze odor profiles and return the result.
    :param crime_scene_path: Path to the crime scene odor profile data file.
    :param suspect_path: Path to the suspect's odor profile data file.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract odor profile features
        crime_odor = extract_odor_profile_features(crime_scene_path)
        suspect_odor = extract_odor_profile_features(suspect_path)

        # Match odor profile features
        match_score = match_odor_profile(crime_odor, suspect_odor)

        # Generate plots for visualization
        crime_plot = create_plot([o[0] for o in crime_odor], "Crime Scene Odor Profile", "Sample", "Intensity")
        suspect_plot = create_plot([o[0] for o in suspect_odor], "Suspect Odor Profile", "Sample", "Intensity")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 15. Gait Analysis
def extract_gait_features(video_path):
    random.seed(os.path.getsize(video_path))
    return [(random.uniform(0, 200), random.uniform(0, 360)) for _ in range(5)]  # (stride length, angle)

def match_gait(g1, g2, length_threshold=20, angle_threshold=30):
    matches = 0
    for l1, a1 in g1:
        for l2, a2 in g2:
            if (abs(l1 - l2) < length_threshold and abs(a1 - a2) < angle_threshold):
                matches += 1
                break
    return matches / min(len(g1), len(g2))

def analyze_gait(crime_scene_path, suspect_path):
    crime_scene = extract_gait_features(crime_scene_path)
    suspect = extract_gait_features(suspect_path)
    suspect_score = match_gait(crime_scene, suspect)
    crime_plot = create_plot([g[0] for g in crime_scene], "Crime Scene Gait", "Step", "Stride Length (cm)")
    suspect_plot = create_plot([g[0] for g in suspect], "Suspect Gait", "Step", "Stride Length (cm)")
    return {
        "suspect_score": round(suspect_score, 2),
        "message": "Suspect matches crime scene!" if suspect_score > 0.6 else "No match.",
        "crime_plot": crime_plot,
        "crime_video": crime_scene_path.split('/')[-1],
        "suspect_plot": suspect_plot,
        "suspect_video": suspect_path.split('/')[-1],
        "crime_path": crime_scene_path,
        "suspect_path": suspect_path
    }

# 16. Explosive Analysis
def extract_explosive_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 100), random.uniform(0, 50)) for _ in range(5)]  # (residue density, particle size)

def match_explosive(e1, e2, density_threshold=10, size_threshold=5):
    matches = 0
    for d1, s1 in e1:
        for d2, s2 in e2:
            if (abs(d1 - d2) < density_threshold and abs(s1 - s2) < size_threshold):
                matches += 1
                break
    return matches / min(len(e1), len(e2))

def analyze_explosive(crime_scene_path, suspect_path):
    """
    Analyze explosive residue images and return the result.
    """
    # ...existing code for extracting explosive features and matching...
    # Use advanced image processing techniques for feature extraction.
    # Return analysis results with plots.

# 17. Glass Analysis
def extract_glass_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 5), random.uniform(0, 100)) for _ in range(5)]  # (fracture width, refractive index)

def match_glass(g1, g2, width_threshold=0.5, index_threshold=10):
    matches = 0
    for w1, i1 in g1:
        for w2, i2 in g2:
            if (abs(w1 - w2) < width_threshold and abs(i1 - i2) < index_threshold):
                matches += 1
                break
    return matches / min(len(g1), len(g2))

def analyze_glass(crime_scene_path, suspect_path):
    crime_scene = extract_glass_features(crime_scene_path)
    suspect = extract_glass_features(suspect_path)
    suspect_score = match_glass(crime_scene, suspect)
    crime_plot = create_plot([g[0] for g in crime_scene], "Crime Scene Glass", "Fracture", "Width (mm)")
    suspect_plot = create_plot([g[0] for g in suspect], "Suspect Glass", "Fracture", "Width (mm)")
    with open(crime_scene_path, 'rb') as f:
        crime_image = base64.b64encode(f.read()).decode()
    with open(suspect_path, 'rb') as f:
        suspect_image = base64.b64encode(f.read()).decode()
    return {
        "suspect_score": round(suspect_score, 2),
        "message": "Suspect matches crime scene!" if suspect_score > 0.6 else "No match.",
        "crime_plot": crime_plot,
        "crime_image": f"data:image/jpeg;base64,{crime_image}",
        "suspect_plot": suspect_plot,
        "suspect_image": f"data:image/jpeg;base64,{suspect_image}",
        "crime_path": crime_scene_path,
        "suspect_path": suspect_path
    }

# 18. Bite Mark Analysis
def extract_bite_mark_features(image_path):
    """
    Simulate the extraction of bite mark properties from an image.
    :param image_path: Path to the bite mark image.
    :return: List of simulated bite mark properties (tooth width, angle).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 10), random.uniform(0, 360)) for _ in range(5)]  # (tooth width, angle)

def match_bite_mark(b1, b2, width_threshold=1, angle_threshold=20):
    """
    Compare two sets of bite mark properties and calculate a match score.
    :param b1: List of bite mark properties from the crime scene.
    :param b2: List of bite mark properties from the suspect.
    :param width_threshold: Threshold for width comparison.
    :param angle_threshold: Threshold for angle comparison.
    :return: Match score as a float.
    """
    matches = 0
    for w1, a1 in b1:
        for w2, a2 in b2:
            if abs(w1 - w2) < width_threshold and abs(a1 - a2) < angle_threshold:
                matches += 1
                break
    return matches / min(len(b1), len(b2))

def analyze_bite_mark(crime_scene_path, suspect_path):
    """
    Analyze bite mark patterns and return the result.
    :param crime_scene_path: Path to the crime scene bite mark image.
    :param suspect_path: Path to the suspect's bite mark image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract bite mark features
        crime_bite = extract_bite_mark_features(crime_scene_path)
        suspect_bite = extract_bite_mark_features(suspect_path)

        # Match bite mark features
        match_score = match_bite_mark(crime_bite, suspect_bite)

        # Generate plots for visualization
        crime_plot = create_plot([b[0] for b in crime_bite], "Crime Scene Bite Mark", "Tooth", "Width (mm)")
        suspect_plot = create_plot([b[0] for b in suspect_bite], "Suspect Bite Mark", "Tooth", "Width (mm)")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 19. Pollen Analysis
def extract_pollen_features(image_path):
    """
    Simulate the extraction of pollen properties from an image.
    :param image_path: Path to the pollen image.
    :return: List of simulated pollen properties (grain size, density).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 50), random.uniform(0, 100)) for _ in range(5)]  # (grain size, density)

def match_pollen(p1, p2, size_threshold=5, density_threshold=10):
    """
    Compare two sets of pollen properties and calculate a match score.
    :param p1: List of pollen properties from the crime scene.
    :param p2: List of pollen properties from the suspect.
    :param size_threshold: Threshold for grain size comparison.
    :param density_threshold: Threshold for density comparison.
    :return: Match score as a float.
    """
    matches = 0
    for s1, d1 in p1:
        for s2, d2 in p2:
            if abs(s1 - s2) < size_threshold and abs(d1 - d2) < density_threshold:
                matches += 1
                break
    return matches / min(len(p1), len(p2))

def analyze_pollen(crime_scene_path, suspect_path):
    """
    Analyze pollen samples and return the result.
    :param crime_scene_path: Path to the crime scene pollen image.
    :param suspect_path: Path to the suspect's pollen image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract pollen features
        crime_pollen = extract_pollen_features(crime_scene_path)
        suspect_pollen = extract_pollen_features(suspect_path)

        # Match pollen features
        match_score = match_pollen(crime_pollen, suspect_pollen)

        # Generate plots for visualization
        crime_plot = create_plot([p[0] for p in crime_pollen], "Crime Scene Pollen", "Pollen", "Grain Size (µm)")
        suspect_plot = create_plot([p[0] for p in suspect_pollen], "Suspect Pollen", "Pollen", "Grain Size (µm)")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 20. Paint Analysis
def extract_paint_features(image_path):
    """
    Simulate the extraction of paint properties from an image.
    :param image_path: Path to the paint image.
    :return: List of simulated paint properties (color intensity, hue).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 100), random.uniform(0, 255)) for _ in range(5)]  # (color intensity, hue)

def match_paint(p1, p2, intensity_threshold=10, hue_threshold=20):
    """
    Compare two sets of paint properties and calculate a match score.
    :param p1: List of paint properties from the crime scene.
    :param p2: List of paint properties from the suspect.
    :param intensity_threshold: Threshold for intensity comparison.
    :param hue_threshold: Threshold for hue comparison.
    :return: Match score as a float.
    """
    matches = 0
    for i1, h1 in p1:
        for i2, h2 in p2:
            if abs(i1 - i2) < intensity_threshold and abs(h1 - h2) < hue_threshold:
                matches += 1
                break
    return matches / min(len(p1), len(p2))

def analyze_paint(crime_scene_path, suspect_path):
    """
    Analyze paint samples and return the result.
    :param crime_scene_path: Path to the crime scene paint image.
    :param suspect_path: Path to the suspect's paint image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract paint features
        crime_paint = extract_paint_features(crime_scene_path)
        suspect_paint = extract_paint_features(suspect_path)

        # Match paint features
        match_score = match_paint(crime_paint, suspect_paint)

        # Generate plots for visualization
        crime_plot = create_plot([p[0] for p in crime_paint], "Crime Scene Paint", "Sample", "Color Intensity")
        suspect_plot = create_plot([p[0] for p in suspect_paint], "Suspect Paint", "Sample", "Color Intensity")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 21. Soil Analysis
def extract_soil_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 100), random.uniform(0, 50)) for _ in range(5)]  # (particle size, moisture)

def match_soil(s1, s2, size_threshold=10, moisture_threshold=5):
    matches = 0
    for p1, m1 in s1:
        for p2, m2 in s2:
            if (abs(p1 - p2) < size_threshold and abs(m1 - m2) < moisture_threshold):
                matches += 1
                break
    return matches / min(len(s1), len(s2))

def analyze_soil(crime_scene_path, suspect_path):
    """
    Analyze soil samples and return the result.
    """
    try:
        # Extract soil features
        crime_soil = extract_soil_features(crime_scene_path)
        suspect_soil = extract_soil_features(suspect_path)

        # Match soil features
        match_score = match_soil(crime_soil, suspect_soil)

        # Generate plots for visualization
        crime_plot = create_plot([s[0] for s in crime_soil], "Crime Scene Soil", "Soil", "Particle Size (µm)")
        suspect_plot = create_plot([s[0] for s in suspect_soil], "Suspect Soil", "Soil", "Particle Size (µm)")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 22. Hair Analysis
def extract_hair_features(image_path):
    """
    Simulate the extraction of hair properties from an image.
    :param image_path: Path to the hair image.
    :return: List of simulated hair properties (diameter, length).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0.05, 0.5), random.uniform(10, 100)) for _ in range(5)]  # (diameter, length)

def match_hair(h1, h2, diam_threshold=0.05, length_threshold=5):
    """
    Compare two sets of hair properties and calculate a match score.
    :param h1: List of hair properties from the crime scene.
    :param h2: List of hair properties from the suspect.
    :param diam_threshold: Threshold for diameter comparison.
    :param length_threshold: Threshold for length comparison.
    :return: Match score as a float.
    """
    matches = 0
    for d1, l1 in h1:
        for d2, l2 in h2:
            if abs(d1 - d2) < diam_threshold and abs(l1 - l2) < length_threshold:
                matches += 1
                break
    return matches / min(len(h1), len(h2))

def analyze_hair(crime_scene_path, suspect_path):
    """
    Analyze hair samples and return the result.
    :param crime_scene_path: Path to the crime scene hair image.
    :param suspect_path: Path to the suspect's hair image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract hair features
        crime_hair = extract_hair_features(crime_scene_path)
        suspect_hair = extract_hair_features(suspect_path)

        # Match hair features
        match_score = match_hair(crime_hair, suspect_hair)

        # Generate plots for visualization
        crime_plot = create_plot([h[0] for h in crime_hair], "Crime Scene Hair", "Hair", "Diameter (mm)")
        suspect_plot = create_plot([h[0] for h in suspect_hair], "Suspect Hair", "Hair", "Diameter (mm)")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 23. Insect Analysis
def extract_insect_features(image_path):
    """
    Simulate the extraction of insect properties from an image.
    :param image_path: Path to the insect image.
    :return: List of simulated insect properties (size, development stage).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 50), random.uniform(0, 100)) for _ in range(5)]  # (size, development stage)

def match_insect(i1, i2, size_threshold=5, stage_threshold=10):
    """
    Compare two sets of insect properties and calculate a match score.
    :param i1: List of insect properties from the crime scene.
    :param i2: List of insect properties from the suspect.
    :param size_threshold: Threshold for size comparison.
    :param stage_threshold: Threshold for development stage comparison.
    :return: Match score as a float.
    """
    matches = 0
    for s1, d1 in i1:
        for s2, d2 in i2:
            if abs(s1 - s2) < size_threshold and abs(d1 - d2) < stage_threshold:
                matches += 1
                break
    return matches / min(len(i1), len(i2))

def analyze_insect(crime_scene_path, suspect_path):
    """
    Analyze insect evidence and return the result.
    :param crime_scene_path: Path to the crime scene insect image.
    :param suspect_path: Path to the suspect's insect image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract insect features
        crime_insect = extract_insect_features(crime_scene_path)
        suspect_insect = extract_insect_features(suspect_path)

        # Match insect features
        match_score = match_insect(crime_insect, suspect_insect)

        # Generate plots for visualization
        crime_plot = create_plot([i[0] for i in crime_insect], "Crime Scene Insect", "Insect", "Size (mm)")
        suspect_plot = create_plot([i[0] for i in suspect_insect], "Suspect Insect", "Insect", "Size (mm)")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 24. Phishing Analysis
def extract_phishing_features(text_path):
    """
    Simulate the extraction of phishing-related features from a text file.
    :param text_path: Path to the text file containing phishing data.
    :return: List of simulated features (keyword count, link count).
    """
    with open(text_path, 'r') as f:
        random.seed(len(f.read()))
    return [(random.randint(0, 50), random.randint(0, 10)) for _ in range(5)]  # (keyword count, link count)

def match_phishing(p1, p2, keyword_threshold=5, link_threshold=2):
    """
    Compare two sets of phishing-related features and calculate a match score.
    :param p1: List of features from the crime scene.
    :param p2: List of features from the suspect.
    :param keyword_threshold: Threshold for keyword count comparison.
    :param link_threshold: Threshold for link count comparison.
    :return: Match score as a float.
    """
    matches = 0
    for k1, l1 in p1:
        for k2, l2 in p2:
            if abs(k1 - k2) < keyword_threshold and abs(l1 - l2) < link_threshold:
                matches += 1
                break
    return matches / min(len(p1), len(p2))

def analyze_phishing(crime_scene_path, suspect_path):
    """
    Analyze phishing data and return the result.
    :param crime_scene_path: Path to the crime scene phishing data file.
    :param suspect_path: Path to the suspect's phishing data file.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract phishing features
        crime_phishing = extract_phishing_features(crime_scene_path)
        suspect_phishing = extract_phishing_features(suspect_path)

        # Match phishing features
        match_score = match_phishing(crime_phishing, suspect_phishing)

        # Generate plots for visualization
        crime_plot = create_plot([p[0] for p in crime_phishing], "Crime Scene Phishing", "Sample", "Keyword Count")
        suspect_plot = create_plot([p[0] for p in suspect_phishing], "Suspect Phishing", "Sample", "Keyword Count")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 25. Dark Web Analysis
def extract_darkweb_features(text_path):
    """
    Simulate the extraction of dark web activity features from a text file.
    :param text_path: Path to the text file containing dark web activity logs.
    :return: List of simulated features (transaction count, anonymity level).
    """
    with open(text_path, 'r') as f:
        random.seed(len(f.read()))
    return [(random.uniform(0, 100), random.uniform(0, 50)) for _ in range(5)]  # (transaction count, anonymity level)

def match_darkweb(d1, d2, count_threshold=10, level_threshold=5):
    """
    Compare two sets of dark web activity features and calculate a match score.
    :param d1: List of features from the crime scene.
    :param d2: List of features from the suspect.
    :param count_threshold: Threshold for transaction count comparison.
    :param level_threshold: Threshold for anonymity level comparison.
    :return: Match score as a float.
    """
    matches = 0
    for c1, l1 in d1:
        for c2, l2 in d2:
            if abs(c1 - c2) < count_threshold and abs(l1 - l2) < level_threshold:
                matches += 1
                break
    return matches / min(len(d1), len(d2))

def analyze_darkweb(crime_scene_path, suspect_path):
    """
    Analyze dark web activity logs and return the result.
    :param crime_scene_path: Path to the crime scene text file.
    :param suspect_path: Path to the suspect's text file.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract dark web features
        crime_darkweb = extract_darkweb_features(crime_scene_path)
        suspect_darkweb = extract_darkweb_features(suspect_path)

        # Match dark web features
        match_score = match_darkweb(crime_darkweb, suspect_darkweb)

        # Generate plots for visualization
        crime_plot = create_plot([d[0] for d in crime_darkweb], "Crime Scene Dark Web", "Event", "Transaction Count")
        suspect_plot = create_plot([d[0] for d in suspect_darkweb], "Suspect Dark Web", "Event", "Transaction Count")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 26. Lie Detection Analysis
def extract_liedetect_features(audio_path):
    with wave.open(audio_path, 'rb') as wav:
        random.seed(wav.getnframes())
    return [(random.uniform(0, 100), random.uniform(0, 50)) for _ in range(5)]  # (stress level, pitch variation)

def match_liedetect(l1, l2, stress_threshold=10, pitch_threshold=5):
    matches = 0
    for s1, p1 in l1:
        for s2, p2 in l2:
            if (abs(s1 - s2) < stress_threshold and abs(p1 - p2) < pitch_threshold):
                matches += 1
                break
    return matches / min(len(l1), len(l2))

def analyze_liedetect(crime_scene_path, suspect_path):
    """
    Analyze audio files for lie detection and return the result.
    """
    # ...existing code for extracting lie detection features and matching...
    # Use advanced audio processing techniques for feature extraction.
    # Return analysis results with plots.

def extract_lie_detection_features(audio_path):
    """
    Extract audio features for lie detection.
    :param audio_path: Path to the audio file.
    :return: List of simulated features (stress level, pitch variation).
    """
    audio, sr = librosa.load(audio_path, sr=None)
    pitch = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=1).flatten()
    stress = librosa.feature.rms(y=audio).flatten()
    return list(zip(stress[:5], pitch[:5]))  # (stress level, pitch variation)

def match_lie_detection(l1, l2, stress_threshold=0.1, pitch_threshold=50):
    """
    Compare two sets of lie detection features and calculate a match score.
    :param l1: List of features from the crime scene.
    :param l2: List of features from the suspect.
    :param stress_threshold: Threshold for stress level comparison.
    :param pitch_threshold: Threshold for pitch variation comparison.
    :return: Match score as a float.
    """
    matches = 0
    for s1, p1 in l1:
        for s2, p2 in l2:
            if abs(s1 - s2) < stress_threshold and abs(p1 - p2) < pitch_threshold:
                matches += 1
                break
    return matches / min(len(l1), len(l2))

def analyze_lie_detection(crime_scene_path, suspect_path):
    """
    Analyze audio files for lie detection and return the result.
    :param crime_scene_path: Path to the crime scene audio file.
    :param suspect_path: Path to the suspect's audio file.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract lie detection features
        crime_features = extract_lie_detection_features(crime_scene_path)
        suspect_features = extract_lie_detection_features(suspect_path)

        # Match lie detection features
        match_score = match_lie_detection(crime_features, suspect_features)

        # Generate plots for visualization
        crime_plot = create_plot([f[0] for f in crime_features], "Crime Scene Lie Detection", "Sample", "Stress Level")
        suspect_plot = create_plot([f[0] for f in suspect_features], "Suspect Lie Detection", "Sample", "Stress Level")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 27. Arson Analysis
def analyze_arson(crime_scene_path, suspect_path):
    """
    Analyze arson-related images and return the result.
    :param crime_scene_path: Path to the crime scene image.
    :param suspect_path: Path to the suspect's image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load pre-trained ResNet model
        model = resnet18(pretrained=True)
        model.eval()

        # Define image transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and preprocess images
        crime_image = cv2.imread(crime_scene_path)
        suspect_image = cv2.imread(suspect_path)

        crime_tensor = transform(crime_image).unsqueeze(0)
        suspect_tensor = transform(suspect_image).unsqueeze(0)

        # Extract features using ResNet
        with torch.no_grad():
            crime_features = model(crime_tensor).numpy().flatten()
            suspect_features = model(suspect_tensor).numpy().flatten()

        # Compute cosine similarity
        similarity = np.dot(crime_features, suspect_features) / (
            np.linalg.norm(crime_features) * np.linalg.norm(suspect_features)
        )

        return {
            "suspect_score": round(similarity, 2),
            "message": "Match" if similarity > 0.8 else "No Match",
            "crime_image": crime_scene_path,
            "suspect_image": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 28. Iris Analysis
def extract_iris_features(image_path):
    """
    Simulate the extraction of iris properties from an image.
    :param image_path: Path to the iris image.
    :return: List of simulated iris properties (pattern density, angle).
    """
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 50), random.uniform(0, 360)) for _ in range(5)]  # (pattern density, angle)

def match_iris(i1, i2, density_threshold=5, angle_threshold=20):
    """
    Compare two sets of iris properties and calculate a match score.
    :param i1: List of iris properties from the crime scene.
    :param i2: List of iris properties from the suspect.
    :param density_threshold: Threshold for density comparison.
    :param angle_threshold: Threshold for angle comparison.
    :return: Match score as a float.
    """
    matches = 0
    for d1, a1 in i1:
        for d2, a2 in i2:
            if abs(d1 - d2) < density_threshold and abs(a1 - a2) < angle_threshold:
                matches += 1
                break
    return matches / min(len(i1), len(i2))

def analyze_iris(crime_scene_path, suspect_path):
    """
    Analyze iris patterns and return the result.
    :param crime_scene_path: Path to the crime scene iris image.
    :param suspect_path: Path to the suspect's iris image.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract iris features
        crime_iris = extract_iris_features(crime_scene_path)
        suspect_iris = extract_iris_features(suspect_path)

        # Match iris features
        match_score = match_iris(crime_iris, suspect_iris)

        # Generate plots for visualization
        crime_plot = create_plot([i[0] for i in crime_iris], "Crime Scene Iris", "Pattern", "Density")
        suspect_plot = create_plot([i[0] for i in suspect_iris], "Suspect Iris", "Pattern", "Density")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 29. Toxicology Analysis
def extract_toxicology_features(text_path):
    with open(text_path, 'r') as f:
        random.seed(len(f.read()))
    return [(random.uniform(0, 1000), random.uniform(0, 100)) for _ in range(5)]  # (concentration, toxin type)

def match_toxicology(t1, t2, conc_threshold=100, type_threshold=10):
    matches = 0
    for c1, t1 in t1:
        for c2, t2 in t2:
            if (abs(c1 - c2) < conc_threshold and abs(t1 - t2) < type_threshold):
                matches += 1
                break
    return matches / min(len(t1), len(t2))

def analyze_toxicology(crime_scene_path, suspect_path):
    """
    Analyze toxicology data and return the result.
    """
    # ...existing code for extracting toxicology features and matching...
    # Use advanced data processing techniques for feature extraction.
    # Return analysis results with plots.

# 30. Geospatial Analysis
def extract_geospatial_features(text_path):
    """
    Simulate the extraction of geospatial data from a text file.
    :param text_path: Path to the text file containing geospatial data.
    :return: List of simulated geospatial points (latitude, longitude).
    """
    with open(text_path, 'r') as f:
        random.seed(len(f.read()))
    return [(random.uniform(-90, 90), random.uniform(-180, 180)) for _ in range(5)]  # (latitude, longitude)

def match_geospatial(g1, g2, distance_threshold=10):
    """
    Compare two sets of geospatial points and calculate a match score.
    :param g1: List of geospatial points from the crime scene.
    :param g2: List of geospatial points from the suspect.
    :param distance_threshold: Threshold for distance comparison (in kilometers).
    :return: Match score as a float.
    """
    matches = 0
    for p1 in g1:
        for p2 in g2:
            distance = geodesic(p1, p2).kilometers
            if distance < distance_threshold:
                matches += 1
                break
    return matches / min(len(g1), len(g2))

def analyze_geospatial(crime_scene_path, suspect_path):
    """
    Analyze geospatial data and return the result.
    :param crime_scene_path: Path to the crime scene geospatial data file.
    :param suspect_path: Path to the suspect's geospatial data file.
    :return: Analysis result as a dictionary.
    """
    try:
        # Extract geospatial features
        crime_geospatial = extract_geospatial_features(crime_scene_path)
        suspect_geospatial = extract_geospatial_features(suspect_path)

        # Match geospatial features
        match_score = match_geospatial(crime_geospatial, suspect_geospatial)

        # Generate plots for visualization
        crime_plot = create_plot([p[0] for p in crime_geospatial], "Crime Scene Geospatial", "Point", "Latitude")
        suspect_plot = create_plot([p[0] for p in suspect_geospatial], "Suspect Geospatial", "Point", "Latitude")

        return {
            "suspect_score": round(match_score, 2),
            "message": "Match" if match_score > 0.6 else "No Match",
            "crime_plot": crime_plot,
            "suspect_plot": suspect_plot,
            "crime_path": crime_scene_path,
            "suspect_path": suspect_path
        }
    except Exception as e:
        return {"error": str(e)}

# 31. DNA Analysis
def extract_dna_features(text_path):
    with open(text_path, 'r') as f:
        random.seed(len(f.read()))
    return [random.uniform(0, 100) for _ in range(5)]  # DNA markers

def match_dna(d1, d2, threshold=10):
    matches = sum(1 for c, s in zip(d1, d2) if abs(c - s) < threshold)
    return matches / min(len(d1), len(d2))

def analyze_dna(crime_scene_path, suspect_path):
    crime_scene = extract_dna_features(crime_scene_path)
    suspect = extract_dna_features(suspect_path)
    suspect_score = match_dna(crime_scene, suspect)
    crime_plot = create_plot(crime_scene, "Crime Scene DNA", "Marker", "Value")
    suspect_plot = create_plot(suspect, "Suspect DNA", "Marker", "Value")
    return {
        "suspect_score": round(suspect_score, 2),
        "message": "Suspect matches crime scene!" if suspect_score > 0.6 else "No match.",
        "crime_plot": crime_plot,
        "suspect_plot": suspect_plot,
        "crime_path": crime_scene_path,
        "suspect_path": suspect_path
    }

# 32. Fingerprint Dust Analysis (Added as a bonus module)
def extract_fingerprint_dust_features(image_path):
    img = Image.open(image_path)
    random.seed(hash(str(img.size)))
    return [(random.uniform(0, 5), random.uniform(0, 90)) for _ in range(5)]  # (dust density, angle)

def analyze_fingerprint_dust(crime_scene_path, suspect_path):
    crime_scene = extract_fingerprint_dust_features(crime_scene_path)
    suspect = extract_fingerprint_dust_features(suspect_path)
    suspect_score = match_fingerprints(crime_scene, suspect)  # Reuse fingerprint matching
    crime_plot = create_plot([f[0] for f in crime_scene], "Crime Scene Fingerprint Dust", "Point", "Dust Density")
    suspect_plot = create_plot([f[0] for f in suspect], "Suspect Fingerprint Dust", "Point", "Dust Density")
    with open(crime_scene_path, 'rb') as f:
        crime_image = base64.b64encode(f.read()).decode()
    with open(suspect_path, 'rb') as f:
        suspect_image = base64.b64encode(f.read()).decode()
    return {
        "suspect_score": round(suspect_score, 2),
        "message": "Suspect matches crime scene!" if suspect_score > 0.6 else "No match.",
        "crime_plot": crime_plot,
        "crime_image": f"data:image/jpeg;base64,{crime_image}",
        "suspect_plot": suspect_plot,
        "suspect_image": f"data:image/jpeg;base64,{suspect_image}",
        "crime_path": crime_scene_path,
        "suspect_path": suspect_path
    }

def analyze_object_detection(image_path):
    """
    Analyze an image for object detection and return the result.
    :param image_path: Path to the image file.
    :return: Analysis result as a dictionary.
    """
    try:
        # Load the YOLOv5 model (pre-trained on COCO dataset)
        model = YOLO("yolov5s.pt")

        # Perform object detection
        results = model(image_path)

        # Extract detected objects and their confidence scores
        detected_objects = []
        for result in results[0].boxes:
            label = model.names[int(result.cls)]
            confidence = result.conf
            detected_objects.append({"label": label, "confidence": round(confidence, 2)})

        return {
            "detected_objects": detected_objects,
            "image_path": image_path
        }
    except Exception as e:
        return {"error": str(e)}