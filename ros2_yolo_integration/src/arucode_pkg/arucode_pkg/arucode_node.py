import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import cv2.aruco as aruco
import numpy as np
from std_msgs.msg import String


class ArucoDetector(Node):
    def __init__(self):
        super().__init__("aruco_detector")

        # 訂閱影像
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )

        self.publisher_ = self.create_publisher(String, '/face_aruco', 10)

        # ArUco 字典
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters_create()

        # 相機內參
        self.camera_matrix = np.array([
            [576.83946, 0.0, 319.59192],
            [0.0, 577.82786, 238.89255],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.001750, -0.003776, -0.000528, -0.000228, 0.000000])

        # ArUco 實際邊長 (單位 m)
        self.marker_length = 0.3

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # 偵測 ArUco
        corners, ids, _ = aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None:
            # 姿態估計
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            for i, id in enumerate(ids.flatten()):
                rvec, tvec = rvecs[i], tvecs[i]
                self.get_logger().info(
                    f"ID {id} -> Pos (marker in cam): {tvec.flatten()}, RotVec: {rvec.flatten()}"
                )
                # 取前一個 tvec 來判斷 X
                if len(ids) >= 1:
                    # 已知 rvec
                    R, _ = cv2.Rodrigues(rvecs[0])  # rotation matrix

                    # ArUco 的 Z 軸 = 法向量（標記的 local Z）
                    marker_normal_in_cam = R @ np.array([0, 0, 1])

                    # 相機看出去是 cam Z = [0, 0, 1]
                    # 我們要讓相機的 Z 與 marker_normal 平行 → 其實就是要讓 yaw 調整到正對

                    # 投影到 XY 平面看方向
                    yaw_error = np.arctan2(marker_normal_in_cam[0], marker_normal_in_cam[2])  # heading error

                    print(f"Marker ID {id} yaw error: {yaw_error:.3f} rad")
                    if yaw_error  > 0.1:
                        rotation_cmd = 'CLOCKWISE_ROTATION'
                    elif yaw_error < -0.1:
                        rotation_cmd = 'COUNTERCLOCKWISE_ROTATION'
                    else:
                        rotation_cmd = 'NO_ROTATION'

                    # 發布到 /face_aruco
                    msg = String()
                    msg.data = rotation_cmd
                    self.publisher_.publish(msg)


            # 如果有至少 2 個 marker，計算正中間
            if len(ids) >= 2:
                midpoint = (tvecs[0].flatten() + tvecs[1].flatten()) / 2.0
                distance = np.linalg.norm(midpoint)
                self.get_logger().info(f"Midpoint (of first 2 markers) in cam frame: {midpoint}, distance: {distance:.3f} m")

                # 你可以依據 midpoint 來判斷相機是否在兩個 marker 正中間 ± 容差
                # 例如：中點 X,Y 要接近 0 => 表示相機在正中間正上方



def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
