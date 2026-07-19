from ultralytics import YOLOE

# Load YOLOE model
model = YOLOE("models/yoloe-11s-seg.pt")

# Define prompt
model.set_classes([
    "person",
    "helmet",
    "safety vest",
    "person without helmet",
    "person without safety vest"
])

# Predict
results = model.predict(
    source="data/input/people-working.jpg",
    conf=0.5
)

results[0].save("data/output/result.jpg")