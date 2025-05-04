# Car Plate Detection with YOLOv8

This project demonstrates how to use the YOLOv8 algorithm to detect and extract car plates from images and videos. Below, you'll find an explanation of YOLO, how to train it (including a shortcut with Roboflow), and the complete code to train the model and process a video to extract car plates.

---

## Understanding the YOLO Algorithm

### What is YOLO?
YOLO (You Only Look Once) is a state-of-the-art object detection algorithm designed for real-time performance. Unlike older methods that scan an image multiple times, YOLO processes the entire image in one pass through a neural network, making it fast and efficient.

### How Does YOLO Work?
1. **Grid Division**: The input image is split into a grid (e.g., 7x7 or finer in newer versions like YOLOv8).
2. **Bounding Box Prediction**:
   - Predicts multiple bounding boxes (rectangles around potential objects).
   - Assigns confidence scores (likelihood of object existence and box accuracy).
   - Determines class probabilities (e.g., "car_plate", "person").
3. **Single Pass**: A convolutional neural network (CNN) processes the image once, outputting predictions for all grid cells simultaneously.
4. **Non-Max Suppression**: Overlapping boxes are filtered to keep only the most confident detections.
5. **Final Output**: Bounding boxes with class labels and confidence scores.

### Why Use YOLO?
✅ **Speed**: Perfect for real-time tasks like video analysis.
✅ **Simplicity**: Combines detection and classification in one step.
✅ **Accuracy**: YOLOv8 balances speed and precision effectively.

---

## Training YOLOv8 for Car Plate Detection

### Traditional Training Process
1. **Collect Images**: Gather images with car plates.
2. **Label Data**: Use tools like LabelImg to annotate images.
3. **Prepare a YAML File**: Create a `dataset.yaml` file specifying:
   ```yaml
   train: ./dataset/train/images
   val: ./dataset/valid/images
   test: ./dataset/test/images
   nc: 1
   names: ['car_plate']
   ```
4. **Train the Model**: Fine-tune a pre-trained YOLO model (e.g., `yolov8n.pt`) on your dataset.
5. **Evaluate & Save**: Save the best weights (e.g., `best.pt`).

### Simplifying Training with Roboflow
Instead of manual dataset preparation, use Roboflow:
- **Find or Upload**: Use Roboflow Universe or upload images.
- **Label**: Annotate images using Roboflow’s interface.
- **Export**: Download in YOLO format, including `data.yaml`.
- **Train**: Use the `data.yaml` in your training script.

---

## Implementation

### 1. Install Dependencies and Train the Model
```python
# Install Ultralytics library
!pip install ultralytics

from ultralytics import YOLO

# Load a pre-trained YOLOv8n model (nano version)
model = YOLO('yolov8n.pt')

# Train the model on your dataset
model.train(data='path/to/dataset.yaml', epochs=50, imgsz=640)
```
- **What it does**: Installs the required library, loads YOLOv8, and trains it for 50 epochs.

### 2. Test the Model on an Image
```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference on a test image
image_path = 'test_image.png'
results = model(image_path)

# Display the results
results[0].show()
```
- **What it does**: Loads the trained model and tests it on an image.

### 3. Extract Car Plates from a Video
```python
from ultralytics import YOLO
import cv2
import os

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Video file path
video_path = 'cars_video.mp4'
cap = cv2.VideoCapture(video_path)

# Set up directory to save cropped plates
output_dir = 'data/cropped_plates/'
os.makedirs(output_dir, exist_ok=True)

plate_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        label = results[0].names[int(box.cls)]
        if label == 'car_plate':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_plate = frame[y1:y2, x1:x2]
            plate_filename = f'{output_dir}plate_{plate_counter}.jpg'
            cv2.imwrite(plate_filename, cropped_plate)
            plate_counter += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Video Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
- **What it does**: Detects car plates in a video, saves cropped plates as images, and displays the video with bounding boxes.

---
## Final Results

https://drive.google.com/file/d/1aIiVoqAmD_2dHU7o4r0e1mHUzHWlG_Ru/view?usp=sharing


## How to Use This Project
### Setup:
1. Install Python and Jupyter Notebook.
2. Ensure you have a dataset (`data.yaml`).
3. Update file paths in the code (`dataset.yaml`, `best.pt`, `cars_video.mp4`).

### Running the Code:
1. Copy the code into a Jupyter Notebook (`car_plate_detection.ipynb`).
2. Execute each cell in order.

---

### Notes:  
- Replace placeholder paths (`path/to/dataset.yaml`, `best.pt`, `cars_video.mp4`) with actual file locations.  
- Ensure the class name (`car_plate`) matches what’s in `data.yaml`.  
- If using Roboflow, download your dataset and point the training script to `data.yaml`.  
- **If you want to extract numbers using OCR, you can use libraries like Tesseract, EasyOCR, or PaddleOCR.**  
- **For accurate text extraction, ensure that the camera used is specifically designed for capturing license plates. It should support high-quality video recording and be capable of capturing vehicles moving at speeds of at least 100 km/h.**  

🚀 **Now you're ready to train and deploy your own car plate detection system!**


Notes
Replace placeholder paths (e.g., 'path/to/dataset.yaml', 'runs/detect/train/weights/best.pt', 'cars_video.mp4') with your actual file locations.
Ensure the class name ('car_plate') matches what’s in your data.yaml file.
If using Roboflow, download your dataset and point the training script to its data.yaml.


## License

This project is licensed under the [MIT License](./LICENSE).
