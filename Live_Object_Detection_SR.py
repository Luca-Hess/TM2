import speech_recognition as sr
from ultralytics import YOLO
import cv2
import Levenshtein
import time
import threading

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

# Function to listen for the "stop" command
def listen_for_stop(stop_event):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while not stop_event.is_set():
            print("Listening for 'stop' command:")
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio)
                if command.lower() == "stop":
                    print("Stop command recognized. Halting task.")
                    stop_event.set()
                    break
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition service.")
                break

# Load the YOLO model
model = YOLO("yolo11n.pt")  # changed model, yolo11n is a lot faster, but struggles with accuracy

# Access all possible class names
class_names = model.names

# Get the target class name from voice input
target_class_name = get_target_class_name()

# Find the closest class name
if target_class_name:
    closest_class_name = find_closest_class_name(target_class_name, class_names)
    print(f"Closest class name found: {closest_class_name}")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("recorded_video.mp4", fourcc, fps, (width, height))

    start_time = time.time()
    detection_time = None
    target_detected = False

    # Create a stop event
    stop_event = threading.Event()

    # Start a thread to listen for the "stop" command
    stop_thread = threading.Thread(target=listen_for_stop, args=(stop_event,))
    stop_thread.start()

    # Loop through the video frames
    while cap.isOpened() and not stop_event.is_set():
        # Read a frame from the webcam
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame and only consider detections with > 50% confidence
            results = model(frame, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Extract and print the center coordinates of detected objects
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name == closest_class_name:
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    print(f"Detected {class_name} at center coordinates: ({center_x}, {center_y})")
                    # Draw a red circle at the center coordinates
                    cv2.circle(annotated_frame, (center_x, center_y), 8, (0, 0, 255), -1)
                    if not target_detected:
                        detection_time = time.time()
                    target_detected = True

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            # Halt the task if no target is detected within 15 seconds
            if time.time() - start_time > 15 and not target_detected:
                print("No target detected within 15 seconds. Halting task.")
                stop_event.set()
                break

            # Keep running for at least 15 more seconds after the target is detected
            if target_detected and detection_time and time.time() - detection_time > 15:
                print("Task completed. Halting task.")
                stop_event.set()
                break
        else:
            # Break the loop if there is an error reading the frame
            stop_event.set()
            break

    # Release the video capture and writer objects and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
else:
    print("Error: Could not recognize a valid class name.")

# hey there