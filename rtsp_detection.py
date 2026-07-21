from ultralytics import YOLO
import cv2

RTSP_URL = "rtsp://132.239.12.145/axis-media/media.amp"

model = YOLO("models/ppe.pt")

results = model.predict(source=RTSP_URL, stream=True, show=False, classes=[0,2,4,5,7])

for r in results:
    # Get the annotated frame (with bounding boxes and labels drawn)
    annotated_frame = r.plot()
    
    # Display the live feed
    cv2.imshow("YOLO RTSP Detection", annotated_frame)
    
    # Press 'q' on the keyboard to exit the stream loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up windows
cv2.destroyAllWindows()