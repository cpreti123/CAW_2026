import cv2
from ultralytics import YOLO

# Load a small YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Running YOLO person detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, verbose=False)

    annotated = frame.copy()

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # COCO class 0 = person
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"person {conf:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Person Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()