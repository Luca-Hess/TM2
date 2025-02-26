from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolo11n.pt") #changed model, yolo11n is a lot faster, but struggles with accuracy

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

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame and only consider detections with > 50% confidence
        results = model(frame, conf=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Extract and print the center coordinates of detected objects
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            print(f"Center coordinates: ({center_x}, {center_y})")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if there is an error reading the frame
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()