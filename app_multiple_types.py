from pathlib import Path

from ultralytics import YOLO

from detector import process_image, process_video

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

model = YOLO("models/ppe.pt")

input_dir = Path("data/input")
image_output = Path("data/output/images")
snapshot_output = Path("data/output/snapshots")

image_output.mkdir(parents=True, exist_ok=True)
snapshot_output.mkdir(parents=True, exist_ok=True)

for file in input_dir.iterdir():

    if file.suffix.lower() in IMAGE_EXT:
        process_image(model, file, image_output)

    elif file.suffix.lower() in VIDEO_EXT:
        process_video(model, file, snapshot_output, cooldown_seconds=3)