from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolo11x.pt")

# Access all possible class names
class_names = model.names

# Define the class name you are looking for
target_class_name = "cup"  # replace with the desired class name

# Verify if the target class name is valid
if target_class_name not in class_names.values():
    print(f"Error: '{target_class_name}' is not a valid class name.")
else:
    # Run batched inference on a list of images
    results = model(["Sample_Screenshot2.jpg"], stream=True)  # return a generator of Results objects

    # Process results generator
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        target_detected = False
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name == target_class_name:
                target_detected = True
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                print(f"Detected {class_name} at center coordinates: ({center_x}, {center_y})")
                # Draw a circle at the center coordinates
                annotated_frame = result.plot()
                cv2.circle(annotated_frame, (center_x, center_y), 8, (0, 0, 255), -1)
                # Save the annotated frame
                cv2.imwrite("result_11x.jpg", annotated_frame)

        if not target_detected:
            print(f"{target_class_name} not detected")
        result.show()  # display to screen