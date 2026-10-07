import os
import cv2
import numpy as np
import requests
import math

# --- 1. Cascade-Datei automatisch herunterladen (falls nicht vorhanden) ---
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{CASCADE_FILENAME}"

if not os.path.exists(CASCADE_FILENAME):
    print("Lade OpenCV Haar-Cascade herunter...")
    resp = requests.get(CASCADE_URL)
    with open(CASCADE_FILENAME, "wb") as f:
        f.write(resp.content)

face_cascade = cv2.CascadeClassifier(CASCADE_FILENAME)

if face_cascade.empty():
    raise RuntimeError("Fehler: Haar-Cascade konnte nicht geladen werden!")

# --- 2. Emojis direkt im Speicher laden (Twemoji CDN) ---
EMOJI_URLS = [
    "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f600.png",  # 😀
    "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f60d.png",  # 😍
    "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f60e.png",  # 😎
    "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f911.png",  # 🤑
    "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f921.png",  # 🤡
]


def load_emoji(url):
    resp = requests.get(url)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


print("Lade Emojis herunter...")
EMOJIS = [load_emoji(url) for url in EMOJI_URLS]


# --- 3. Transparente PNG-Overlays auf Videobild anwenden ---
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    h, w, c = overlay.shape
    if y < 0 or x < 0 or y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    if c == 4:
        alpha = overlay[:, :, 3] / 255.0
        for channel in range(0, 3):
            background[y:y + h, x:x + w, channel] = (
                    alpha * overlay[:, :, channel] +
                    (1.0 - alpha) * background[y:y + h, x:x + w, channel]
            )
    return background


# --- 4. Main Loop & Centroid Tracking ---
tracked_faces = {}
next_face_id = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gesichtserkennung
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    current_detections = []

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        current_detections.append({'box': (x, y, w, h), 'centroid': (cx, cy)})

    # Centroid Tracking
    updated_tracked = {}
    used_detections = set()

    for face_id, data in tracked_faces.items():
        best_dist = float("inf")
        best_idx = -1
        for idx, det in enumerate(current_detections):
            if idx in used_detections:
                continue
            dist = math.hypot(data['centroid'][0] - det['centroid'][0], data['centroid'][1] - det['centroid'][1])
            if dist < best_dist and dist < 120:
                best_dist = dist
                best_idx = idx

        if best_idx != -1:
            used_detections.add(best_idx)
            updated_tracked[face_id] = {
                'centroid': current_detections[best_idx]['centroid'],
                'box': current_detections[best_idx]['box'],
                'emoji_idx': data['emoji_idx'],
                'disappeared': 0
            }
        else:
            data['disappeared'] += 1
            if data['disappeared'] < 10:
                updated_tracked[face_id] = data

    for idx, det in enumerate(current_detections):
        if idx not in used_detections:
            updated_tracked[next_face_id] = {
                'centroid': det['centroid'],
                'box': det['box'],
                'emoji_idx': next_face_id % len(EMOJIS),
                'disappeared': 0
            }
            next_face_id += 1

    tracked_faces = updated_tracked

    # --- 5. Smileys über den Köpfen zeichnen ---
    for face_id, data in tracked_faces.items():
        if 'box' not in data:
            continue
        x, y, bw, bh = data['box']
        emoji_img = EMOJIS[data['emoji_idx']]

        size = int(bw * 0.8)
        emoji_x = x + (bw - size) // 2
        emoji_y = max(0, y - size - 10)

        overlay_transparent(frame, emoji_img, emoji_x, emoji_y, (size, size))

    cv2.imshow("Face Time Emoji Live (OpenCV)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()