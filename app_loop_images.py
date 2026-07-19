import cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO("models/ppe.pt")
print("Model names", model.names)

input_dir = Path("data/input")
output_dir = Path("data/output")

output_dir.mkdir(exist_ok=True)

allowed_classes = {
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest"
}
violation_classes = {
    "NO-Hardhat",
    "NO-Safety Vest"
}
person_count = 0
helmet_count = 0
violation_helmet_count = 0
vest_count = 0
violation_vest_count = 0

for image_path in input_dir.glob("*"):
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    print(f"Processing {image_path.name}")

    # Filter only 0=Hardhat, 2=NO-Hardhat, 4=Safety Vest, 5=NO-Safety Vest, 7=Person
    results = model(str(image_path), classes=[0,2,4,5,7])
    img = cv2.imread(str(image_path))
    for result in results:
        for box in result.boxes:
            cls_name = model.names[int(box.cls)]

            if cls_name == "Person":
                person_count += 1
            elif cls_name == "Hardhat":
                helmet_count += 1
            elif cls_name == "NO-Hardhat":
                violation_helmet_count += 1
            elif cls_name == "Safety Vest":
                vest_count += 1
            elif cls_name == "NO-Safety Vest":
                violation_vest_count += 1

            if cls_name not in violation_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "NO-Hardhat" or cls_name == "NO-Safety Vest":
                color = (0, 0, 255)  # Red for violations
            else:
                color = (0, 255, 0)  # Green for compliant

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cls_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        output_file = output_dir / f"{image_path.stem}-result{image_path.suffix}"
        cv2.imwrite(str(output_file), img)

print("--------------------------------")
print("Detection completed successfully")
print("--------------------------------")
print("Person Count:", person_count)
print("Helmet Count:", helmet_count)
print("Vest Count:", vest_count)
print("x Violation Helmet Count:", violation_helmet_count)
print("x Violation Vest Count:", violation_vest_count)
