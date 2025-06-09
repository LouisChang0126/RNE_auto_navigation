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
        
        # 相機參數
        self.image_width = 640
        self.image_height = 480
        self.camera_matrix = np.array([
            [576.83946, 0.0, 319.59192],
            [0.0, 577.82786, 238.89255],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.001750, -0.003776, -0.000528, -0.000228, 0.000000])
        
        # 初始化YOLO模型
        model_path = os.path.join(
            get_package_share_directory("yolo_example_pkg"), "models", "object_detection_best.pt"
        )
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f'YOLO model loaded successfully from {model_path}')
            self.get_logger().info(f'Available classes: {self.model.names}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model from {model_path}: {str(e)}')
            return
        
        # 初始化CV橋
        self.bridge = CvBridge()
        
        # 訂閱壓縮圖像
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)
        
        # 發布檢測結果
        self.detection_publisher = self.create_publisher(
            String,
            '/yolo_detections',
            10)
        
        # 發布動作建議
        self.action_publisher = self.create_publisher(
            String,
            '/action_suggestion',
            10)
        
        # 新增：發布action_detection
        self.action_detection_publisher = self.create_publisher(
            String,
            '/action_detection',
            10)
        
        self.get_logger().info('YOLO Detection Node has been started')

    def image_callback(self, msg):
        try:
            # 將ROS壓縮圖像消息轉換為OpenCV格式
            self.get_logger().debug(f'Received compressed image, format: {msg.format}')
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().debug(f'Image converted to OpenCV, shape: {cv_image.shape}')
            
            # 相機校正
            undistorted_image = self.undistort_image(cv_image)
            
            # 執行YOLO檢測
            results = self.model(undistorted_image)
            self.get_logger().debug('YOLO detection completed')
            
            # 處理檢測結果
            detections = []
            pikachu_detected = False
            pikachu_x = None
            # 新增：用於action_detection的變數
            pikachu_in_middle_third = False
            pikachu_in_middle_bottom = False
            non_pikachu_in_middle_bottom = False
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 獲取檢測框中心點和尺寸
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    center_x = x / self.image_width  # 正規化到[0,1]
                    center_y = y / self.image_height  # 正規化到[0,1]
                    width = w / self.image_width  # 正規化寬度
                    height = h / self.image_height  # 正規化高度
                    
                    # 將像素坐標轉換到相機坐標系
                    cam_x, cam_y = self.pixel_to_camera_coords(center_x * self.image_width, 
                                                             center_y * self.image_height)
                    
                    # 獲取類別名稱
                    cls_id = int(box.cls)
                    class_name = self.model.names[cls_id]
                    
                    # 創建檢測結果字典
                    detection = {
                        'tag': class_name,
                        'x-axis': float(cam_x),
                        'y-axis': float(cam_y),
                        'width': float(width),
                        'height': float(height)
                    }
                    detections.append(detection)
                    
                    # 檢查是否為Pikachu
                    if class_name.lower() == 'pikachu':
                        pikachu_detected = True
                        pikachu_x = x  # 記錄像素x坐標
                        self.get_logger().info(f'Pikachu detected at x={x}')
                        # 新增：檢查Pikachu是否在中間1/3
                        if self.image_width / 3 <= x <= 2 * self.image_width / 3:
                            pikachu_in_middle_third = True
                            # 檢查是否在最下面1/6
                            if 5 * self.image_height / 6 <= y <= self.image_height:
                                pikachu_in_middle_bottom = True
                    else:
                        # 新增：檢查非Pikachu是否在中間1/3且最下面1/6
                        if (self.image_width / 3 <= x <= 2 * self.image_width / 3 and
                            5 * self.image_height / 6 <= y <= self.image_height):
                            non_pikachu_in_middle_bottom = True
            
            # 發布檢測結果
            if detections:
                detection_msg = String()
                detection_msg.data = json.dumps(detections)
                self.detection_publisher.publish(detection_msg)
                self.get_logger().info(f'Published detections: {detection_msg.data}')
            else:
                self.get_logger().info('No detections found')
            
            # 發布動作建議
            action_msg = String()
            if pikachu_detected and pikachu_x is not None:
                if pikachu_x < self.image_width / 3:
                    action_msg.data = 'left'
                elif pikachu_x > 2 * self.image_width / 3:
                    action_msg.data = 'right'
                else:
                    action_msg.data = 'forward'
                self.get_logger().info(f'Action suggestion: {action_msg.data}')
            else:
                action_msg.data = 'none'
                self.get_logger().info('No Pikachu detected, action: none')
            self.action_publisher.publish(action_msg)
            
            # 新增：發布action_detection
            action_detection_msg = String()
            if pikachu_in_middle_bottom:
                action_detection_msg.data = 'STOP'
                self.get_logger().info('Pikachu in middle 1/3 and bottom 1/6, action: STOP')
            elif non_pikachu_in_middle_bottom:
                action_detection_msg.data = 'ROTATE'
                self.get_logger().info('Non-Pikachu in middle 1/3 and bottom 1/6, action: ROTATE')
            elif pikachu_in_middle_third:
                action_detection_msg.data = 'FORWARD'
                self.get_logger().info('Pikachu in middle 1/3, action: FORWARD')
            else:
                action_detection_msg.data = 'NONE'
                self.get_logger().info('No relevant detections, action: NONE')
            self.action_detection_publisher.publish(action_detection_msg)
                
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
            undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            self.get_logger().debug('Image undistorted successfully')
            return undistorted
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
            self.get_logger().debug(f'Pixel ({px}, {py}) converted to camera coords ({cam_x}, {cam_y})')
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