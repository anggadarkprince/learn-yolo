from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO

# Only display these classes
ALLOWED_CLASSES = {
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
}

# These are considered violations
VIOLATION_CLASSES = {
    "NO-Hardhat",
    "NO-Safety Vest",
}

def process_image(model: YOLO, image_path: Path, output_dir: Path):
    print(f"Processing image: {image_path.name}")

    results = model(image_path, conf=0.8, classes=[0,2,4,5,7])

    for result in results:
        keep = []
        has_violation = False

        for i, box in enumerate(result.boxes):
            cls_name = model.names[int(box.cls)]

            if cls_name == "Person":
                frame = cv2.imread(str(image_path))
                if not is_valid_detection(box, frame):
                    continue

            #if cls_name in ALLOWED_CLASSES:
            #    keep.append(i)

            if cls_name in VIOLATION_CLASSES:
                keep.append(i)
                has_violation = True

        result.boxes = result.boxes[keep]

        annotated = result.plot()

        output_file = (output_dir / f"{image_path.stem}-result{image_path.suffix}")

        cv2.imwrite(str(output_file), annotated)

        print(f"Saved {output_file.name}")

        if has_violation:
            print("Violation detected")


def process_video(model: YOLO, video_path: Path, snapshot_dir: Path, cooldown_seconds: int = 3):
    print(f"Processing video: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    cooldown_frames = int(fps * cooldown_seconds)

    frame_index = 0
    last_saved_frame = -cooldown_frames

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_index += 1
        results = model(frame, verbose=False, conf=0.8, classes=[0,2,4,5,7])
        result = results[0]

        keep = []
        has_violation = False

        for i, box in enumerate(result.boxes):
            cls_name = model.names[int(box.cls)]

            if cls_name == "Person":
                if not is_valid_detection(box, frame):
                    continue

            #if cls_name in ALLOWED_CLASSES:
            #    keep.append(i)

            if cls_name in VIOLATION_CLASSES:
                keep.append(i)
                has_violation = True

        result.boxes = result.boxes[keep]

        if has_violation:
            if frame_index - last_saved_frame >= cooldown_frames:
                annotated = result.plot()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                filename = (
                    f"{video_path.stem}"
                    f"_frame{frame_index}"
                    f"_{timestamp}.jpg"
                )

                cv2.imwrite(str(snapshot_dir / filename), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                print(f"Violation snapshot saved ({frame_index})")

                last_saved_frame = frame_index

    cap.release()


def is_valid_detection(box, frame):
    # Minimal confidence threshold
    if box.conf < 0.80:
        return False

    x1, y1, x2, y2 = box.xyxy[0]
    width = x2 - x1
    height = y2 - y1

    # Skip small boxes (likely false positives)
    if width < 80:
        return False

    if height < 150:
        return False

    # Skip boxes that are too close to the edges of the frame
    h, w = frame.shape[:2]
    margin = 10
    if (x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin):
        return False

    return True