import speech_recognition as sr
from ultralytics import YOLO
import cv2
import Levenshtein

# Function to get target class name from voice input
def get_target_class_name():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say the target class name:")
        audio = recognizer.listen(source)
        try:
            target_class_name = recognizer.recognize_google(audio)
            print(f"Target class name recognized: {target_class_name}")
            return target_class_name
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
    return None

# Function to find the closest class name
def find_closest_class_name(target_class_name, class_names):
    closest_class_name = min(class_names.values(), key=lambda name: Levenshtein.distance(target_class_name, name))
    return closest_class_name

# Load the YOLO model
model = YOLO("yolo11x.pt")

# Access all possible class names
class_names = model.names

# Get the target class name from voice input
target_class_name = get_target_class_name()

# Find the closest class name
if target_class_name:
    closest_class_name = find_closest_class_name(target_class_name, class_names)
    print(f"Closest class name found: {closest_class_name}")

    # Run batched inference on a list of images
    results = model(["Sample_Screenshot2.jpg"], stream=True)  # return a generator of Results objects

    # Process results generator
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        target_detected = False
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name == closest_class_name:
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
            print(f"{closest_class_name} not detected")
        result.show()  # display to screen
else:
    print("Error: Could not recognize a valid class name.")