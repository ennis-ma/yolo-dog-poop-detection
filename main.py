
import cv2
from ultralytics import YOLO

# Load YOLOv11s (downloads automatically if not cached)
model = YOLO("yolo11n.pt")

# Open your RTSP stream
rtsp_url = "rtsp://192.168.1.194:8556/g2h"
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

frame_skip = 10
frame_count = 0

while True:   
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Run YOLO detection
    results = model(frame, imgsz=320)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show video
    cv2.imshow("YOLOv11s RTSP Feed", annotated_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
