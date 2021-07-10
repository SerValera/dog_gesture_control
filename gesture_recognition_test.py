import cv2
from src.hand_tracker import HandTracker
import numpy as np
from tensorflow.keras.models import load_model

# ------------
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"
detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)
# ------------

sign_classifier = load_model('models/model2.h5')

SIGNS = ['one', 'two', 'three', 'four', 'five', 'ok', 'rock', 'thumbs_up']
SIGNS_dict = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'ok': 6,
    'rock': 7,
    'thumbs_up': 8
}

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 3

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]



while True:
    hasFrame, frame = capture.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    frame = cv2.flip(frame, 1)

    points, _ = detector(image)

    if points is not None:
        sign_coords = points.flatten() / float(frame.shape[0]) - 0.5
        sign_class = sign_classifier.predict(np.expand_dims(sign_coords, axis=0))
        sign_text = SIGNS[sign_class.argmax()]

        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        wrist_x = int(points[0][0])
        wrist_y = int(points[0][1])
        cv2.putText(frame, sign_text, (wrist_x - 20, wrist_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

        gesture_ml = int(SIGNS_dict[sign_text])

    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


capture.release()
cv2.destroyAllWindows()
