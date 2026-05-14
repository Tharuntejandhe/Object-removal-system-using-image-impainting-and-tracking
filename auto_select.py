import cv2
import json
import os
import numpy as np

frame_path = "frames/000001.png"
if not os.path.exists(frame_path):
    print("Frame not found!")
    exit(1)

img = cv2.imread(frame_path)
h, w, _ = img.shape

# Make a box in the middle. Let's make it fairly large, 30% of the screen size
box_w = int(w * 0.3)
box_h = int(h * 0.3)
cx, cy = w // 2, h // 2

x1 = cx - box_w // 2
y1 = cy - box_h // 2
x2 = cx + box_w // 2
y2 = cy + box_h // 2

selection = {
  "mode": "box",
  "frame": frame_path,
  "box": {
    "x1": x1,
    "y1": y1,
    "x2": x2,
    "y2": y2
  }
}

os.makedirs("config", exist_ok=True)
with open("config/selection.json", "w") as f:
    json.dump(selection, f, indent=2)

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
os.makedirs("output", exist_ok=True)
preview_path = "output/selection_preview.png"
cv2.imwrite(preview_path, img)

mask = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
os.makedirs("masks_raw", exist_ok=True)
cv2.imwrite("masks_raw/000001.png", mask)

print(f"Generated auto-selection: {x1},{y1} to {x2},{y2}")
