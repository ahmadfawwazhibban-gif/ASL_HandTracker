# 🤟 Real-Time ASL Hand Tracker & Translator

**Project Overview**
This project focuses on leveraging computer vision and spatial tracking to develop a real-time American Sign Language (ASL) gesture recognition tool. 
The project follows a live data processing workflow, including camera initialization, 3D hand landmark extraction, finger state evaluation, gesture classification, and dynamic UI rendering.

**Supported Gestures:**
- ✋ Hello!
- 🖐️ Thank You
- 🤟 I Love You!
- 🫵 You?
- 👍 How are (you)
- ✊ Goodbye!

### 🎯 Project Objectives
1. Extract and analyze real-time 3D hand landmarks using webcam feeds.
2. Investigate and map the relationship between individual finger extensions and overall hand orientation.
3. Develop a robust depth-based (X, Y, Z) pinch-to-toggle system to activate/deactivate translation without false positives from 2D occlusion.
4. Implement debounce timers and grace counters to eliminate UI flickering during rapid hand movements.
5. Render dynamic, spatial text overlays that anchor intelligently to specific finger groupings based on the recognized gesture.

### 📂 Feature & Landmark Description
Instead of a static dataset, this project processes live feed data using MediaPipe's 21 3D hand landmarks.

| Feature | Description |
| :--- | :--- |
| **Landmarks 1-4** | Thumb joints (used for distance tracking and toggle triggers) |
| **Landmarks 5-8** | Index finger joints (used for extension states and pinch detection) |
| **Landmarks 9-20** | Middle, Ring, and Pinky finger joints (used for gesture classification) |
| **Wrist (0)** | Base anchor for calculating hand tilt and height thresholds |
| **3D Distance (Z-axis)** | Depth tracking to differentiate true pinches from overlapping camera angles |

### 🛠️ Technologies and Libraries Used
- **Python** – Core programming language
- **OpenCV (cv2)** – Video capture, image processing, and UI drawing
- **MediaPipe** – 3D hand landmark detection and tracking
- **NumPy** – Array operations and image copying
- **Math** – Hypotenuse and 3D distance calculations
- **Time** – Debounce tracking and gesture hold timers

### ⚙️ Project Workflow

**1. Initialization & Preprocessing**
- Loaded the MediaPipe `hand_landmarker.task` model.
- Initialized camera feeds and flipped frames horizontally for a natural mirror effect.
- Converted BGR frames to RGB for MediaPipe processing.

**2. Hand Tracking & Extraction**
- Detected up to 2 simultaneous hands in the frame.
- Extracted 21 spatial landmarks per hand.
- Mapped relationships between fingertip and PIP/DIP joints to evaluate open/closed states.

**3. Gesture Classification Logic**
- Evaluated finger extensions (Index, Middle, Ring, Pinky).
- Calculated thumb distance and hand tilt (wrist vs. middle tip).
- Passed states through a conditional dictionary to classify specific ASL signs.

**4. State Management & Debouncing**
- Implemented a 1-second hold timer before committing to a gesture display.
- Added a 3-frame grace counter to remember gestures if tracking briefly drops.
- Required a 0.4-second deliberate hold on the thumb-index pinch to toggle the system on/off.

**5. Dynamic UI Rendering**
- Anchored text boxes dynamically (e.g., floating between the index and pinky for "I Love You", or above the knuckles for "Goodbye").
- Restricted UI coordinates to keep text boxes strictly within the screen bounds.

### 📊 Key Observations
1. Standard 2D distance calculations create false positives when the hand rotates; incorporating the Z-axis is strictly necessary for accurate toggle triggers.
2. "Hello" and "Thank You" share identical finger states; horizontal hand tilt and vertical wrist thresholds successfully differentiate them.
3. Rapid micro-movements cause flickering; implementing dictionary-based state tracking completely stabilizes the visual output.
4. Anchoring text dynamically to the active fingers creates a highly intuitive, augmented-reality feel compared to static subtitles.

### 🏁 Conclusion
This project demonstrates the application of real-time computer vision techniques to accessibility and gesture classification. 
The script successfully balances strict geometric thresholds with forgiving grace timers to create a fluid, user-friendly experience. 
The project provided practical experience in spatial mathematics, state management, UI rendering, and live video processing.

### ⚠️ Limitations
- **Lighting & Background:** The model's accuracy degrades in low light or against highly cluttered backgrounds.
- **Occlusion:** Fingers completely hidden from the camera view may occasionally be misclassified by the underlying MediaPipe model.
- **Static Vocabulary:** The tool currently recognizes a hardcoded dictionary of signs and does not interpret full ASL grammar, facial expressions, or motion-heavy continuous sentences.
- **Hardware Dependency:** Performance and frame rate rely directly on the user's CPU and webcam quality.

### 👨‍💻 Author
**Ahmad Fawwaz Hibban**
Computer Vision | Python | OpenCV

⭐ If you find this project useful, feel free to explore the repository and share your feedback.
