from ultralytics import YOLO
import cv2
import numpy as np

# 모델 로드
model = YOLO("best.pt")

# 영상 로드
cap = cv2.VideoCapture("test_video.mp4")

# 점수
score = {"home": 0, "away": 0}

# 골인 판정 변수
ball_prev_y = None

# 3점 라인 포인트
three_point_pts = []
three_point_ready = False


def mouse_click(event, x, y, flags, param):
    """마우스 클릭으로 3점 라인 포인트 설정"""
    global three_point_pts, three_point_ready
    if event == cv2.EVENT_LBUTTONDOWN:
        three_point_pts.append((x, y))
        print(f"Point added: ({x}, {y}) - Total: {len(three_point_pts)}")
        if len(three_point_pts) >= 5:
            three_point_ready = True
            print("3점 라인 설정 완료! 아무 키나 누르세요.")


def is_outside_three_point(ball_x, ball_y, pts):
    """공이 3점 라인 밖에 있는지 판단"""
    if len(pts) < 3:
        return False
    polygon = np.array(pts, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon, (ball_x, ball_y), False)
    return result < 0  # 음수면 폴리곤 밖 = 3점


def detect_goal(ball_x, ball_y, rim_x, rim_y, rim_width, ball_prev_y):
    """공이 림을 위에서 아래로 통과했는지 판정"""
    if ball_prev_y is None:
        return False
    if abs(ball_x - rim_x) < rim_width // 2:
        if ball_prev_y < rim_y and ball_y > rim_y:
            return True
    return False


# --- 3점 라인 설정 단계 ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 800)
ret, first_frame = cap.read()
if not ret:
    print("영상을 불러올 수 없습니다.")
    exit()

cv2.namedWindow("CourtEyes - 3점 라인 설정")
cv2.setMouseCallback("CourtEyes - 3점 라인 설정", mouse_click)

print("3점 라인을 따라 5개 이상 클릭하세요. 완료되면 아무 키나 누르세요.")

while not three_point_ready:
    display = first_frame.copy()
    cv2.putText(
        display,
        f"Click 3-point line ({len(three_point_pts)} / min 5)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    for pt in three_point_pts:
        cv2.circle(display, pt, 6, (0, 255, 255), -1)

    if len(three_point_pts) >= 2:
        for i in range(len(three_point_pts) - 1):
            cv2.line(
                display, three_point_pts[i], three_point_pts[i + 1], (0, 255, 255), 2
            )

    cv2.imshow("CourtEyes - 3점 라인 설정", display)
    key = cv2.waitKey(1)
    if key != -1 and three_point_ready:
        break

cv2.destroyAllWindows()

# --- 메인 루프 ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame, persist=True, verbose=False, conf=0.15, imgsz=1280, device="mps"
    )

    ball_box = None
    rim_box = None
    ball_y = None

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls == 0:  # ball
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

            elif cls == 2:  # rim
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

    # 3점 라인 표시
    if len(three_point_pts) >= 2:
        pts_array = np.array(three_point_pts, dtype=np.int32)
        cv2.polylines(frame, [pts_array], False, (0, 255, 255), 2)

    # 골인 판정
    if ball_box and rim_box:
        ball_x, ball_y = ball_box
        rim_x, rim_y, rim_width = rim_box
        print(f"ball: ({ball_x}, {ball_y}) rim: ({rim_x}, {rim_y}) width: {rim_width}")
        if detect_goal(ball_x, ball_y, rim_x, rim_y, rim_width, ball_prev_y):
            outside = is_outside_three_point(ball_x, ball_y, three_point_pts)
            points = 3 if outside else 2
            score["home"] += points
            print(f"GOAL! {points}점! Score: {score}")

    ball_prev_y = ball_y

    # 점수 표시
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
