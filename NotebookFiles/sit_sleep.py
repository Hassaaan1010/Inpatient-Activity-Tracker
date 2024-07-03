import cv2
import mediapipe as mp
import numpy as np


# Define the calculateAngle function
def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# CAMERA MEDIA PIPING
lying = Stopwatch(2)
sitting = Stopwatch(2)
state = None  # can be "standing", "sitting", "lying"
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("Mediapipe feed: ", frame)
        # time.sleep(1)
        if True:
            # recolor image to rgb from cv2 default (bgr)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # make detection and store in result
            results = pose.process(image)

            # convert back to cv2 default bgr
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            ]

            angle = int(calculateAngle(shoulder, hip, ankle))

            cv2.putText(
                image,
                str(angle),
                tuple(np.multiply(hip, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Error: {e}")
            continue

        if state != "lying" and angle > 140:
            lying.start()
            sitting.stop()
            state = "lying"

        if state == "lying" and angle < 140:
            sitting.start()
            lying.stop()
            state = "sitting"

        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        cv2.putText(
            image,
            "ANGLE",
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(angle),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            "STATE",
            (65, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            state,
            (60, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mp_drawing.draw_landmarks(
            image,  # output
            results.pose_landmarks,  # passing landmarks
            mp_pose.POSE_CONNECTIONS,  # passing landmark connections
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        cv2.imshow("Mediapipe feed: ", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
