import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YOLOVideoNode(Node):
    def __init__(self):
        super().__init__('yolo_video_node')
        self.subscription = self.create_subscription(
            Image,
            'video_topic',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.model = YOLO("yolo11n.pt")
        self.out = cv2.VideoWriter("recorded_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, conf=0.5)
        annotated_frame = results[0].plot()
        self.out.write(annotated_frame)
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.out.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    yolo_video_node = YOLOVideoNode()
    rclpy.spin(yolo_video_node)
    yolo_video_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()