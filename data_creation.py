# print("STARTING PROGRAM")
# import cv2
# print("cv2 imported")
# import mediapipe as mp
# print("mediapipe imported")
# import numpy as np
# import os

# print("Starting dataset creator...")

# # Camera init (Windows fix)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("Camera failed")
# else:
#     print("Camera working")

# input("Press Enter to exit...")
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()
# print("Initializing MediaPipe...")
# # MediaPipe init
# try:
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(max_num_hands=1)
#     print("MediaPipe initialized")
# except Exception as e:
#     print("MediaPipe FAILED:", e)
#     exit()
# # mp_hands = mp.solutions.hands
# # hands = mp_hands.Hands(max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

# canvas = np.zeros((480, 640, 3), dtype=np.uint8)
# points = []

# DATASET_PATH = "dataset"
# os.makedirs(DATASET_PATH, exist_ok=True)

# label = "A"
# count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Frame not received")
#         break

#     frame = cv2.flip(frame, 1)

#     try:
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb)
#     except Exception as e:
#         print("MediaPipe error:", e)
#         break

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             h, w, _ = frame.shape

#             x = int(handLms.landmark[8].x * w)
#             y = int(handLms.landmark[8].y * h)

#             points.append((x, y))

#             # Draw lines
#             for i in range(1, len(points)):
#                 cv2.line(canvas, points[i-1], points[i], (255,255,255), 5)

#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#     # Combine
#     combined = cv2.add(frame, canvas)

#     cv2.putText(combined, f"Label: {label}", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.imshow("Dataset Creator", combined)

#     key = cv2.waitKey(1) & 0xFF

#     # Change label
#     if 65 <= key <= 90:
#         label = chr(key)
#         print("Label:", label)

#     # Save
#     elif key == ord('s'):
#         path = os.path.join(DATASET_PATH, label)
#         os.makedirs(path, exist_ok=True)

#         img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (28,28))

#         filename = f"{label}_{count}.png"
#         cv2.imwrite(os.path.join(path, filename), img)

#         print("Saved:", filename)
#         count += 1

#     # Clear
#     elif key == ord('c'):
#         canvas = np.zeros((480, 640, 3), dtype=np.uint8)
#         points = []

#     # Quit
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# print("STARTING PROGRAM")

# import cv2
# print("cv2 imported")

# import mediapipe as mp
# print("mediapipe imported")

# import numpy as np
# import os

# print("Starting dataset creator...")

# # Camera init
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     print("Camera failed")
#     exit()
# else:
#     print("Camera working")

# print("Initializing MediaPipe...")

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

# # Canvas
# canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# # Drawing variables
# points = []
# drawing = False
# prev_x, prev_y = 0, 0

# # Dataset
# DATASET_PATH = "dataset"
# os.makedirs(DATASET_PATH, exist_ok=True)

# label = "A"
# count = 0

# print("Press:")
# print("S → Save | C → Clear | Q → Quit | A-Z → Change Label")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Frame not received")
#         break

#     frame = cv2.flip(frame, 1)

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:

#             h, w, _ = frame.shape

#             x = int(handLms.landmark[8].x * w)
#             y = int(handLms.landmark[8].y * h)

#             # ----------- DRAWING LOGIC (KEY PART) -----------
#             # Index finger up → draw
#             if handLms.landmark[8].y < handLms.landmark[6].y:
#                 drawing = True

#                 # Smoothing
#                 alpha = 0.7
#                 x = int(alpha * x + (1 - alpha) * prev_x)
#                 y = int(alpha * y + (1 - alpha) * prev_y)

#                 points.append((x, y))
#                 prev_x, prev_y = x, y

#             else:
#                 drawing = False
#                 points.append(None)  # break stroke
#             # ------------------------------------------------

#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#     # Draw strokes
#     for i in range(1, len(points)):
#         if points[i-1] is None or points[i] is None:
#             continue
#         cv2.line(canvas, points[i-1], points[i], (255, 255, 255), 5)

#     # Combine
#     combined = cv2.add(frame, canvas)

#     cv2.putText(combined, f"Label: {label}", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.imshow("Dataset Creator", combined)

#     key = cv2.waitKey(1) & 0xFF

#     # Change label
#     if 65 <= key <= 90:
#         label = chr(key)
#         print("Label:", label)

#     # Save image
#     elif key == ord('s'):
#         path = os.path.join(DATASET_PATH, label)
#         os.makedirs(path, exist_ok=True)

#         img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (28, 28))

#         filename = f"{label}_{count}.png"
#         cv2.imwrite(os.path.join(path, filename), img)

#         print("Saved:", filename)
#         count += 1

#     # Clear canvas
#     elif key == ord('c'):
#         canvas = np.zeros((480, 640, 3), dtype=np.uint8)
#         points = []
#         print("Canvas cleared")

#     # Quit
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# print("STARTING PROGRAM")
# import cv2
# print("cv2 imported")
# import mediapipe as mp
# print("mediapipe imported")
# import numpy as np
# import os

# print("Starting dataset creator...")

# # Camera init (Windows fix)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("Camera failed")
# else:
#     print("Camera working")

# input("Press Enter to exit...")
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()
# print("Initializing MediaPipe...")
# # MediaPipe init
# try:
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(max_num_hands=1)
#     print("MediaPipe initialized")
# except Exception as e:
#     print("MediaPipe FAILED:", e)
#     exit()
# # mp_hands = mp.solutions.hands
# # hands = mp_hands.Hands(max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

# canvas = np.zeros((480, 640, 3), dtype=np.uint8)
# points = []

# DATASET_PATH = "dataset"
# os.makedirs(DATASET_PATH, exist_ok=True)

# label = "A"
# count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Frame not received")
#         break

#     frame = cv2.flip(frame, 1)

#     try:
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb)
#     except Exception as e:
#         print("MediaPipe error:", e)
#         break

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             h, w, _ = frame.shape

#             x = int(handLms.landmark[8].x * w)
#             y = int(handLms.landmark[8].y * h)

#             points.append((x, y))

#             # Draw lines
#             for i in range(1, len(points)):
#                 cv2.line(canvas, points[i-1], points[i], (255,255,255), 5)

#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#     # Combine
#     combined = cv2.add(frame, canvas)

#     cv2.putText(combined, f"Label: {label}", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.imshow("Dataset Creator", combined)

#     key = cv2.waitKey(1) & 0xFF

#     # Change label
#     if 65 <= key <= 90:
#         label = chr(key)
#         print("Label:", label)

#     # Save
#     elif key == ord('s'):
#         path = os.path.join(DATASET_PATH, label)
#         os.makedirs(path, exist_ok=True)

#         img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (28,28))

#         filename = f"{label}_{count}.png"
#         cv2.imwrite(os.path.join(path, filename), img)

#         print("Saved:", filename)
#         count += 1

#     # Clear
#     elif key == ord('c'):
#         canvas = np.zeros((480, 640, 3), dtype=np.uint8)
#         points = []

#     # Quit
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


print("STARTING PROGRAM")

import cv2
print("cv2 imported")

import mediapipe as mp
print("mediapipe imported")

import numpy as np
import os
def create_clean_image(points, width=640, height=480):
    canvas = np.zeros((64, 64), dtype=np.uint8)

    # Normalize points to 64x64
    norm_points = []
    for p in points:
        if p is None:
            norm_points.append(None)
        else:
            x = int(p[0] / width * 64)
            y = int(p[1] / height * 64)
            norm_points.append((x, y))

    # Draw clean strokes
    for i in range(1, len(norm_points)):
        if norm_points[i-1] is None or norm_points[i] is None:
            continue
        cv2.line(canvas, norm_points[i-1], norm_points[i], 255, 3)

    return canvas


def center_image(img):
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img[y_min:y_max+1, x_min:x_max+1]

    new_img = np.zeros((64, 64), dtype=np.uint8)

    h, w = cropped.shape
    y_offset = (64 - h)//2
    x_offset = (64 - w)//2

    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    return new_img

print("Starting dataset creator...")

# Camera init
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed")
    exit()
else:
    print("Camera working")

print("Initializing MediaPipe...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Drawing variables
points = []
drawing = False
prev_x, prev_y = 0, 0

# Dataset
DATASET_PATH = "dataset1"
os.makedirs(DATASET_PATH, exist_ok=True)

label = "A"
count = 0

print("Press:")
print("S → Save | C → Clear | Q → Quit | A-Z → Change Label")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            h, w, _ = frame.shape

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # ----------- DRAWING LOGIC (FIXED) -----------
            if handLms.landmark[8].y < handLms.landmark[6].y - 0.02:
                drawing = True

                # Stronger smoothing
                alpha = 0.3
                x = int(alpha * x + (1 - alpha) * prev_x)
                y = int(alpha * y + (1 - alpha) * prev_y)

                # Movement threshold to remove jitter
                dx = abs(x - prev_x)
                dy = abs(y - prev_y)

                if dx + dy > 5:
                    points.append((x, y))
                    prev_x, prev_y = x, y

                    # Fill gaps
                    cv2.circle(canvas, (x, y), 8, (255,255,255), -1)
            else:
                drawing = False
                points.append(None)
            # ---------------------------------------------

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Draw strokes
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(canvas, points[i-1], points[i], (255, 255, 255), 12)

    # Combine
    combined = cv2.add(frame, canvas)

    cv2.putText(combined, f"Label: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Dataset Creator", combined)

    key = cv2.waitKey(1) & 0xFF

    # Change label
    if 65 <= key <= 90:
        label = chr(key)
        print("Label:", label)

    # Save image
    elif key == ord('s'):
        path = os.path.join(DATASET_PATH, label)
        os.makedirs(path, exist_ok=True)

        # STEP 1: create clean stroke image
        img = create_clean_image(points)

        # STEP 2: center it
        img = center_image(img)

        # STEP 3: resize for model
        img = cv2.resize(img, (28, 28))

        # STEP 4: OPTIONAL slight dilation (clean thickness)
        img = cv2.dilate(img, np.ones((2,2), np.uint8), iterations=1)

        filename = f"{label}_{count}.png"
        cv2.imwrite(os.path.join(path, filename), img)

        print("Saved CLEAN:", filename)
        count += 1
    # Clear canvas
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        points = []
        print("Canvas cleared")

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()