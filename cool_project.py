import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=2
)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

gesture_start_times = {}
gesture_grace_counters = {}
gesture_current_words = {}

asl_translation_active = True
last_toggle_time = 0.0
pinch_start_time = 0.0 


def get_finger_states(landmarks):
    index_open = landmarks[8].y < landmarks[6].y
    middle_open = landmarks[12].y < landmarks[10].y
    ring_open = landmarks[16].y < landmarks[14].y
    pinky_open = landmarks[20].y < landmarks[18].y

    thumb_dist = math.hypot(landmarks[4].x - landmarks[17].x, landmarks[4].y - landmarks[17].y)

    return index_open, middle_open, ring_open, pinky_open, thumb_dist

def recognize_gesture(landmarks):
    index_open, middle_open, ring_open, pinky_open, thumb_dist = get_finger_states(landmarks)
    
    wrist = landmarks[0]
    middle_tip = landmarks[12]

    if index_open and middle_open and ring_open and pinky_open:
        tilt = abs(middle_tip.x - wrist.x)
        if wrist.y < 0.60 or tilt > 0.15:
            return "Hello!"
        else:
            return "Thank You"

    if index_open and not middle_open and not ring_open and pinky_open and thumb_dist > 0.18:
        return "I Love You!"

    if index_open and not middle_open and not ring_open and not pinky_open:
        return "You?"

    if not index_open and not middle_open and not ring_open and not pinky_open and thumb_dist > 0.18:
        return "How are"

    if not index_open and not middle_open and not ring_open and not pinky_open:
        return "Goodbye!"

    return None

def draw_landmarks(img, detection_result):
    global gesture_start_times, gesture_grace_counters, gesture_current_words
    global asl_translation_active, last_toggle_time, pinch_start_time

    img_copy = np.copy(img)
    height, width, _ = img_copy.shape
    active_indices = []
    detected_candidates = []
    current_time = time.time()

    status_text = "ASL: ON" if asl_translation_active else "ASL: OFF"
    status_color = (0, 255, 0) if asl_translation_active else (0, 0, 255)
    cv2.putText(img_copy, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    any_pinch_detected = False

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        active_indices.append(idx)

        mp_drawing.draw_landmarks(
            img_copy,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        
        pinch_dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )

        if pinch_dist < 0.05:
            any_pinch_detected = True

        if asl_translation_active:
            word = recognize_gesture(hand_landmarks)

            if word:
                gesture_grace_counters[idx] = 0

                if idx not in gesture_start_times or gesture_current_words.get(idx) != word:
                    gesture_start_times[idx] = time.time()
                    gesture_current_words[idx] = word

                elapsed_time = time.time() - gesture_start_times[idx]

                if elapsed_time >= 0.75:
                    if word in ("Hello!", "Thank You"):
                        tips = (8, 12, 16, 20)
                        avg_x = sum(hand_landmarks[t].x for t in tips) / 4.0
                        top_y = min(hand_landmarks[t].y for t in tips)
                        text_x = int(avg_x * width) - 85
                        text_y = int(top_y * height) - 20

                    elif word == "I Love You!":
                        avg_x = (hand_landmarks[8].x + hand_landmarks[20].x) / 2.0
                        top_y = min(hand_landmarks[8].y, hand_landmarks[20].y)
                        text_x = int(avg_x * width) - 100
                        text_y = int(top_y * height) - 20

                    elif word == "How are": 
                        anchor = hand_landmarks[4]
                        text_x = int(anchor.x * width) - 40
                        text_y = int(anchor.y * height) - 20

                    elif word == "Goodbye!":
                        anchor = hand_landmarks[9]
                        text_x = int(anchor.x * width) - 85
                        text_y = int(anchor.y * height) - 40

                    else:
                        anchor = hand_landmarks[12]
                        text_x = int(anchor.x * width) - 40
                        text_y = int(anchor.y * height) - 20

                    text_x = max(10, min(text_x, width - 230))
                    text_y = max(40, min(text_y, height - 20))

                    detected_candidates.append((elapsed_time, word, text_x, text_y))

            else:
                current_grace = gesture_grace_counters.get(idx, 0)
                if current_grace < 3 and idx in gesture_start_times:
                    gesture_grace_counters[idx] = current_grace + 1
                else:
                    gesture_start_times.pop(idx, None)
                    gesture_grace_counters.pop(idx, None)
                    gesture_current_words.pop(idx, None)
        else:
            if idx in gesture_start_times:
                gesture_start_times.pop(idx, None)
                gesture_grace_counters.pop(idx, None)
                gesture_current_words.pop(idx, None)

    if any_pinch_detected:
        if pinch_start_time == 0.0:
            pinch_start_time = current_time
        elif current_time - pinch_start_time > 0.4:  
            if current_time - last_toggle_time > 1.5:  
                asl_translation_active = not asl_translation_active
                last_toggle_time = current_time
                pinch_start_time = 0.0  
                
                gesture_start_times.clear()
                gesture_grace_counters.clear()
                gesture_current_words.clear()
    else:
        pinch_start_time = 0.0

    for stored_idx in list(gesture_start_times.keys()):
        if stored_idx not in active_indices:
            gesture_start_times.pop(stored_idx, None)
            gesture_grace_counters.pop(stored_idx, None)
            gesture_current_words.pop(stored_idx, None)

    if detected_candidates and asl_translation_active:
        if any(word == "How are" for _, word, _, _ in detected_candidates):
            detected_candidates.sort(key=lambda x: x[2], reverse=True)
        else:
            detected_candidates.sort(key=lambda x: x[0], reverse=True)

        _, primary_word, text_x, text_y = detected_candidates[0]

        cv2.rectangle(img_copy, (text_x - 10, text_y - 30), (text_x + 220, text_y + 15), (0, 0, 0), -1)
        cv2.putText(
            img_copy,
            primary_word,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            2,
        )

    return img_copy

def detect_hands(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = detector.detect(img)
    drawn_image = draw_landmarks(img.numpy_view(), detection_result)

    return cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)

        img = detect_hands(frame)
        cv2.imshow('ASL Translator', img)

        if cv2.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()