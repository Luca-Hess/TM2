from ultralytics import YOLO
import cv2

###############################
# OBJECT RECOGNITION IN IMAGE #
###############################

# Load a model
model = YOLO("yolo11x.pt")  # pretrained YOLO model (11n for speed, alternatives: 11s, 11m, 11l or 11x)

# Run batched inference on a list of images
results = model(["Sample_Screenshot2.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result_11x.jpg")  # save to disk


###############################
# OBJECT RECOGNITION IN VIDEO #
###############################

# # Define path to video file
# source = "car-detection.mp4"
#
# # Run inference on the source
# results = model(source, stream=True, conf=0.5)  # generator of Results objects
#
# # Open the video file
# cap = cv2.VideoCapture(source)
#
# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("annotated_result.mp4", fourcc, fps, (width, height))
#
# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#
#     # Annotate frame
#     annotated_frame = result.plot()
#
#     # Write the annotated frame to the output video
#     out.write(annotated_frame)
#
# # Release the video capture and writer objects
# cap.release()
# out.release()


##############################################
# OBJECT RECOGNITION IN VIDEO with live feed #
##############################################

# # Load the YOLO model
# model = YOLO("yolo11n.pt")
#
# # Open the video file
# video_path = "car-detection.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLO inference on the frame
#         results = model(frame, conf=0.5)
#
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLO Inference", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()