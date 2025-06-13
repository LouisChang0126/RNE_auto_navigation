import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
from std_msgs.msg import String
import os
from ament_index_python.packages import get_package_share_directory
import json

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        self.image_width = 640
        self.image_height = 480
        self.camera_matrix = np.array([
            [576.83946, 0.0, 319.59192],
            [0.0, 577.82786, 238.89255],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.001750, -0.003776, -0.000528, -0.000228, 0.000000])

        model_path = os.path.join(
            get_package_share_directory("yolo_example_pkg"), "models", "door_random_best.pt"
        )
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f'YOLO model loaded successfully from {model_path}')
            self.get_logger().info(f'Available classes: {self.model.names}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model from {model_path}: {str(e)}')
            return

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)

        self.action_detection_publisher = self.create_publisher(
            String,
            '/action_detection',
            10)
        
        self.door_detection_publisher = self.create_publisher(
            String,
            '/door_detection',
            10)

        self.get_logger().info('YOLO Detection Node has been started')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            undistorted_image = self.undistort_image(cv_image)
            results = self.model(undistorted_image)

            pikachu_in_middle_third = False
            pikachu_in_left_third = False
            pikachu_in_right_third = False
            pikachu_large_enough = False
            door_in_middle_third = False
            door_in_left_third = False
            door_in_right_third = False

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    center_x = x / self.image_width
                    center_y = y / self.image_height

                    cls_id = int(box.cls)
                    class_name = self.model.names[cls_id]

                    if class_name.lower() == 'pikachu':
                        if self.image_width / 3 <= x <= 2 * self.image_width / 3:
                            pikachu_in_middle_third = True
                        elif x < self.image_width / 3:
                            pikachu_in_left_third = True
                        elif x > 2 * self.image_width / 3:
                            pikachu_in_right_third = True
                        if w * h > 0.6 * self.image_width * self.image_height:
                            pikachu_large_enough = True

                    elif class_name.lower() == 'door':
                        if self.image_width / 3 <= x <= 2 * self.image_width / 3:
                            door_in_middle_third = True
                        elif x < self.image_width / 3:
                            door_in_left_third = True
                        elif x > 2 * self.image_width / 3:
                            door_in_right_third = True

            action_detection_msg = String()
            if pikachu_large_enough:
                action_detection_msg.data = 'STOP'
                self.get_logger().info('Pikachu in bottom 1/2, action: STOP')
            elif pikachu_in_middle_third:
                action_detection_msg.data = 'FORWARD'
                self.get_logger().info('Pikachu in middle 1/3, action: FORWARD')
            elif pikachu_in_left_third:
                action_detection_msg.data = 'COUNTERCLOCKWISE_ROTATION'
                self.get_logger().info('Pikachu in left 1/3, action: COUNTERCLOCKWISE_ROTATION')
            elif pikachu_in_right_third:
                action_detection_msg.data = 'CLOCKWISE_ROTATION'
                self.get_logger().info('Pikachu in right 1/3, action: CLOCKWISE_ROTATION')
            else:
                action_detection_msg.data = 'NONE'
                self.get_logger().info('No relevant detections, action: NONE')
            self.action_detection_publisher.publish(action_detection_msg)

            door_detection_msg = String()
            if door_in_middle_third:
                door_detection_msg.data = 'FORWARD'
                self.get_logger().info('Door detected in middle 1/3')
            elif door_in_left_third:
                door_detection_msg.data = 'COUNTERCLOCKWISE_ROTATION'
                self.get_logger().info('Door in left 1/3, action: COUNTERCLOCKWISE_ROTATION')
            elif door_in_right_third:
                door_detection_msg.data = 'CLOCKWISE_ROTATION'
                self.get_logger().info('Door in right 1/3, action: CLOCKWISE_ROTATION')
            else:
                door_detection_msg.data = 'NONE'
                self.get_logger().info('No door detected in middle 1/3')
            self.door_detection_publisher.publish(door_detection_msg)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error converting compressed image: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def undistort_image(self, image):
        try:
            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.camera_matrix,
                (self.image_width, self.image_height),
                cv2.CV_16SC2
            )
            return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            self.get_logger().error(f'Error undistorting image: {str(e)}')
            return image

    def pixel_to_camera_coords(self, px, py):
        try:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            cam_x = (px - cx) / fx
            cam_y = (py - cy) / fy
            return cam_x, cam_y
        except Exception as e:
            self.get_logger().error(f'Error converting to camera coords: {str(e)}')
            return 0.0, 0.0

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
