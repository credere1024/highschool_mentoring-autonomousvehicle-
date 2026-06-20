from ultralytics import YOLO
import cv2

# =========================
# 설정
# =========================
MODEL_PATH = "best.pt"
CAMERA_INDEX = 1         # 방금 웹캠 테스트에서 된 번호
CONF_THRESHOLD = 0.4
IMG_SIZE = 640

# =========================
# YOLO 모델 불러오기
# =========================
model = YOLO(MODEL_PATH)

print("모델 클래스 목록:")
print(model.names)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# 메인 반복문
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 인식
    results = model(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )

    result = results[0]

    # 감지된 객체 그리기
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = model.names[cls_id]
        label = f"{class_name} {conf:.2f}"

        # 박스
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 글자 배경
        cv2.rectangle(
            frame,
            (x1, y1 - 25),
            (x1 + len(label) * 12, y1),
            (0, 255, 0),
            -1
        )

        # 글자
        cv2.putText(
            frame,
            label,
            (x1, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # 화면 출력
    cv2.imshow("YOLO Webcam Detection", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
