from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture("test_video.mp4")

score = {"home": 0, "away": 0}
ball_prev_y = None


def detect_goal(ball_x, ball_y, rim_x, rim_y, rim_width, ball_prev_y):
    if ball_prev_y is None:
        return False
    if abs(ball_x - rim_x) < rim_width // 2:
        if ball_prev_y < rim_y and ball_y > rim_y:
            return True
    return False


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    ball_box = None
    rim_box = None
    ball_y = None

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls == 0:
                ball_box = (cx, cy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "ball",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2,
                )
            elif cls == 2:
                rim_box = (cx, cy, x2 - x1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "rim",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    if ball_box and rim_box:
        ball_x, ball_y = ball_box
        rim_x, rim_y, rim_width = rim_box
        if detect_goal(ball_x, ball_y, rim_x, rim_y, rim_width, ball_prev_y):
            score["home"] += 2
            print(f"GOAL! Score: {score}")

    ball_prev_y = ball_y

    cv2.putText(
        frame,
        f"Home: {score['home']}  Away: {score['away']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.imshow("CourtEyes", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
