import cv2
from ultralytics import YOLO

model = YOLO("models/ppe.pt")
print("Model names", model.names)

# Filter only 0=Hardhat, 2=NO-Hardhat, 4=Safety Vest, 5=NO-Safety Vest, 7=Person
results = model("data/input/people-working.jpg", classes=[0,2,4,5,7])

allowed_classes = {
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest"
}
person_count = 0
helmet_count = 0
violation_helmet_count = 0
vest_count = 0
violation_vest_count = 0

for result in results:
    keep_cls = []
    for i, box in enumerate(result.boxes):
        cls = int(box.cls)
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0]
        print(cls_name, round(conf, 2), x1.item(), y1.item(), x2.item(), y2.item())

        if cls_name in allowed_classes:
            keep_cls.append(i)

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

    result.save(filename="data/output/ppe-detection-all.jpg")
    result.boxes = result.boxes[keep_cls]
    annotated = result.plot()
    cv2.imwrite("data/output/ppe-detection.jpg", annotated)

print("--------------------------------")
print("Detection completed successfully")
print("--------------------------------")
print("Person Count:", person_count)
print("Helmet Count:", helmet_count)
print("Vest Count:", vest_count)
print("x Violation Helmet Count:", violation_helmet_count)
print("x Violation Vest Count:", violation_vest_count)
