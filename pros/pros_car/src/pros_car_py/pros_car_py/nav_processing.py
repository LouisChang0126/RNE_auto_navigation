from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math


class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False

        self.state = 'SEARCHING'
        
        self.forward_step = 0
        self.rotation_count = 0
        self.rotation_wise = True  # True: clockwise, False: counterclockwise
        
        # 參數（可根據機器人調整）
        self.rotation_steps = 180  # 完成一圈（360度）所需的步數
        self.forward_steps = 200  # 每次前進的步數

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        # 只抓第一次路径
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"
            else:
                print("Get first path")
                self.index = 0
                self.global_plan_msg = (
                    self.data_processor.get_processed_received_global_plan_no_dynamic()
                )
                self.recordFlag = 1
                action_key = "STOP"

        car_position, car_orientation = self.data_processor.get_processed_amcl_pose()

        goal_position = self.ros_communicator.get_latest_goal()
        target_distance = cal_distance(car_position, goal_position)

        # 抓最近的物標(可調距離)
        target_x, target_y = self.get_next_target_point(car_position)

        if target_x is None or target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 計算角度誤差
        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def check_data_availability(self):
        return (
            self.data_processor.get_processed_received_global_plan_no_dynamic()
            and self.data_processor.get_processed_amcl_pose()
            and self.ros_communicator.get_latest_goal()
        )

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]

    def camera_nav(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])

        action = "STOP"
        limit_distance = 0.7

        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 0.8:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def camera_nav_unity(self):
        """實現導航邏輯，返回動作字串"""
        action_detection = self.data_processor.get_action_detection()

        if action_detection == None:
            return "STOP"
        
        action = "STOP"
        print(f'Current state: {self.state}, Action detection: {action_detection}')

        if action_detection == 'STOP':
            return 'STOP'
        if self.state == 'SEARCHING':
            # 正在旋轉搜尋皮卡丘
            self.rotation_step += 1
            action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
            # print(f'Rotating, step {self.rotation_step}/{self.rotation_steps}')
            
            if action_detection == 'COUNTERCLOCKWISE_ROTATION':
                # 發現皮卡丘在左邊
                self.rotation_wise = False
                self.rotation_step = 0
                self.rotation_count = 0
                action = 'COUNTERCLOCKWISE_ROTATION'
                print('Pikachu detected in left, turning left')
            elif action_detection == 'CLOCKWISE_ROTATION':
                # 發現皮卡丘在右邊
                self.rotation_wise = True
                self.rotation_step = 0
                self.rotation_count = 0
                action = 'CLOCKWISE_ROTATION'
                print('Pikachu detected in right, turning right')
            elif action_detection == 'FORWARD':
                # 發現皮卡丘，進入追蹤狀態
                self.state = 'FOLLOWING'
                self.rotation_step = 0
                self.rotation_count = 0
                action = 'FORWARD_SLOW'
                print('Pikachu detected, switching to FOLLOWING state')
            elif self.rotation_step >= self.rotation_steps:
                # 完成一圈旋轉
                self.rotation_count += 1
                self.rotation_step = 0
                self.state = 'MOVING_FORWARD'
                self.forward_step = 0
                action = 'FORWARD'
                print('No Pikachu detected after rotation, moving forward')

        elif self.state == 'FOLLOWING':
            # 追蹤皮卡丘
            if action_detection == 'STOP':
                action = 'STOP'
                print('Pikachu in middle-bottom, stopping')
                self.state = 'SEARCHING'
                self.rotation_step = 0
                self.rotation_count = 0
            elif action_detection == 'FORWARD':
                action = 'FORWARD_SLOW'
                print('Moving forward towards Pikachu')
            elif action_detection == 'COUNTERCLOCKWISE_ROTATION':
                # 發現皮卡丘在左邊
                self.rotation_wise = False
                action = 'COUNTERCLOCKWISE_ROTATION'
                print('Pikachu detected in left, turning left')
            elif action_detection == 'CLOCKWISE_ROTATION':
                # 發現皮卡丘在右邊
                self.rotation_wise = True
                action = 'CLOCKWISE_ROTATION'
                print('Pikachu detected in right, turning right')
            else:
                # 失去皮卡丘，回到搜尋狀態
                self.state = 'SEARCHING'
                self.rotation_step = 0
                action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
                print('Lost Pikachu, switching to SEARCHING state')

        elif self.state == 'MOVING_FORWARD':
            # 前進一段距離後繼續搜尋
            self.forward_step += 1
            action = 'FORWARD'
            # print(f'Moving forward, step {self.forward_step}/{self.forward_steps}')
            if self.forward_step >= self.forward_steps:
                self.state = 'SEARCHING'
                self.rotation_step = 0
                action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
                print('Finished moving forward, resuming search')

        # 如果收到 ROTATE，重置搜尋
        if action_detection == 'ROTATE' and (self.state == 'FOLLOWING' or self.state == 'MOVING_FORWARD'):
            action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
            print('Non-Pikachu in middle-bottom, stopping')
            self.state = 'SEARCHING'
            self.rotation_step = 0
            self.rotation_count = 0

        return action

    def stop_nav(self):
        return "STOP"

# class Nav2Processing_door_random:
#     def __init__(self, ros_communicator, data_processor):
#         self.ros_communicator = ros_communicator
#         self.data_processor = data_processor
#         self.finishFlag = False
#         self.global_plan_msg = None
#         self.index = 0
#         self.index_length = 0
#         self.recordFlag = 0
#         self.goal_published_flag = False

#         self.state = 'SEARCHING'
        
#         self.X = -8
#         self.Y_step = 0
#         self.rotation_angle = 0
#         self.GO_Y_step = 0
#         self.search_dir = True  # True: clockwise, False: counterclockwise
#         self.x_to_go = 0
        
#         # 參數（可根據機器人調整）
#         self.forward_steps = 170.0  # 每次前進的步數
#         self.right_rotate_angle = 90 / 46
#         self.left_rotate_angle = - 90 / 52

#     def reset_nav_process(self):
#         self.finishFlag = False
#         self.recordFlag = 0
#         self.goal_published_flag = False

#         self.rotation_angle = 0
#         self.GO_Y_step = 0
#         self.search_dir = True
#         self.state = 'SEARCHING'
#         self.x_to_go = 0
#         self.X = -8
#         self.Y_step = 0

#     def finish_nav_process(self):
#         self.finishFlag = True
#         self.recordFlag = 1

#     def get_finish_flag(self):
#         return self.finishFlag


#     def camera_nav_unity(self):
#         """實現導航邏輯，返回動作字串"""
#         action_detection = self.data_processor.get_action_detection() #皮卡丘相關訊息
#         door_detection = self.data_processor.get_door_detection() #門相關訊息

#         if action_detection == None or door_detection == None:
#             if action_detection == None:
#                 print("No action detection data available.")
#             if door_detection == None:
#                 print("No door detection data available.")
#             return "STOP"
        
#         action = "STOP"
#         print(f'Current state: {self.state}, Door: {door_detection}')## Pikachu detection: {action_detection},

#         if action_detection == 'STOP' and self.GO_Y_step >= 3:
#             return 'STOP'
#         if self.state == 'SEARCHING':
#             if self.GO_Y_step >= 3:
#                 #搜尋皮卡丘
#                 action = 'FORWARD_SLOW'
#                 if action_detection == 'COUNTERCLOCKWISE_ROTATION':
#                     # 發現皮卡丘在左邊
#                     self.search_dir = False
#                     self.rotation_angle += self.left_rotate_angle
#                     action = 'COUNTERCLOCKWISE_ROTATION'
#                     print('Pikachu detected in left, turning left')
#                 elif action_detection == 'CLOCKWISE_ROTATION':
#                     # 發現皮卡丘在右邊
#                     self.search_dir = True
#                     self.rotation_angle += self.right_rotate_angle
#                     action = 'CLOCKWISE_ROTATION'
#                     print('Pikachu detected in right, turning right')
#                 elif action_detection == 'FORWARD':
#                     # 發現皮卡丘，進入追蹤狀態
#                     action = 'FORWARD_SLOW'
#                     print('Pikachu detected')
#                 else:
#                     if self.search_dir:
#                         self.rotation_angle += self.right_rotate_angle
#                         action = 'CLOCKWISE_ROTATION'
#                     else:
#                         self.rotation_angle += self.left_rotate_angle
#                         action = 'COUNTERCLOCKWISE_ROTATION'
#                     print('No Pikachu detected')
            
#             else:
#                 # 旋轉搜尋門
#                 if door_detection == 'COUNTERCLOCKWISE_ROTATION':
#                     # 發現門在左邊
#                     self.search_dir = False
#                     self.rotation_angle += self.left_rotate_angle
#                     action = 'COUNTERCLOCKWISE_ROTATION'
#                     print('Door detected in left, turning left')
#                 elif door_detection == 'CLOCKWISE_ROTATION':
#                     # 發現門在右邊
#                     self.search_dir = True
#                     self.rotation_angle += self.right_rotate_angle
#                     action = 'CLOCKWISE_ROTATION'
#                     print('Door detected in right, turning right')
#                 elif door_detection == 'FORWARD':
#                     # 發現門，進入追蹤狀態
#                     # if self.GO_Y_step == 0:
#                     if self.rotation_angle > 45:
#                         self.x_to_go = 1.5 * self.forward_steps
#                     elif self.rotation_angle > 0:
#                         self.x_to_go = 0.5 * self.forward_steps
#                     elif self.rotation_angle < -45:
#                         self.x_to_go = - 1.5 * self.forward_steps
#                     else:
#                         self.x_to_go = - 0.5 * self.forward_steps
#                     # else:
#                     #     if self.rotation_angle > 65:
#                     #         self.x_to_go = 2 * self.forward_steps
#                     #     elif self.rotation_angle > 35:
#                     #         self.x_to_go = 1.2 * self.forward_steps
#                     #     elif self.rotation_angle > 0:
#                     #         self.x_to_go = 1 * self.forward_steps
#                     #     elif self.rotation_angle < -65:
#                     #         self.x_to_go = -3 * self.forward_steps
#                     #     elif self.rotation_angle < -35:
#                     #         self.x_to_go = -2 * self.forward_steps
#                     #     else:
#                     #         self.x_to_go = -1 * self.forward_steps
#                     self.state = 'FACE_LEFT' if self.rotation_angle < 0 else 'FACE_RIGHT'
#                     action = 'STOP'
#                     print('Door detected, switching to FOLLOWING state')
#                 else:
#                     if self.search_dir:
#                         if self.rotation_angle <= 90:
#                             # 旋轉面向右
#                             self.rotation_angle += self.right_rotate_angle
#                             action = 'CLOCKWISE_ROTATION'
#                         else:
#                             # 旋轉面向左
#                             self.search_dir = False
#                             self.rotation_angle += self.left_rotate_angle
#                             action = 'COUNTERCLOCKWISE_ROTATION'
#                     else:
#                         if self.rotation_angle >= -90:
#                             # 旋轉面向左
#                             self.rotation_angle += self.left_rotate_angle
#                             action = 'COUNTERCLOCKWISE_ROTATION'
#                         else:
#                             # 旋轉面向右
#                             self.search_dir = True
#                             self.rotation_angle += self.right_rotate_angle
#                             action = 'CLOCKWISE_ROTATION'

#         elif self.state == 'FACE_LEFT':
#             if self.rotation_angle <= -90:
#                 # 完成面向左
#                 self.state = 'GO_X'
#                 action = 'STOP'
#             else:
#                 self.rotation_angle += self.left_rotate_angle
#                 action = 'COUNTERCLOCKWISE_ROTATION'

#         elif self.state == 'FACE_RIGHT':
#             if self.rotation_angle >= 90:
#                 # 完成面向右
#                 self.state = 'GO_X'
#                 action = 'STOP'
#             else:
#                 self.rotation_angle += self.right_rotate_angle
#                 action = 'CLOCKWISE_ROTATION'

#         elif self.state == 'FACE_FRONT':
#             if self.rotation_angle < 0.5 and self.rotation_angle > -0.5:
#                 # 完成面向前
#                 self.state = 'GO_Y'
#                 action = 'STOP'
#                 self.Y_step = 0
#             elif self.rotation_angle > 0:
#                 self.rotation_angle -= 1
#                 action = 'COUNTERCLOCKWISE_ROTATION'
#             else:
#                 self.rotation_angle += self.right_rotate_angle
#                 action = 'CLOCKWISE_ROTATION'

#         elif self.state == 'GO_X':
#             if self.X == self.x_to_go:
#                 # 完成前進
#                 self.state = 'FACE_FRONT'
#                 action = 'STOP'
#             else:
#                 if self.x_to_go < 0:
#                     self.X -= 1
#                 else:
#                     self.X += 1
#                 action = 'FORWARD'

#         elif self.state == 'GO_Y':
#             if self.Y_step > self.forward_steps * 1.05:
#                 # 完成前進
#                 self.state = 'SEARCHING'
#                 self.GO_Y_step += 1
#                 action = 'STOP'
#             else:
#                 self.Y_step += 1
#                 action = 'FORWARD'
            
#         print(f'Action: {action}, Rotation angle: {self.rotation_angle}, X: {self.X}|{self.x_to_go}')
#         return action

#     def stop_nav(self):
#         return "STOP"

# class Nav2Processing_door_random:
#     def __init__(self, ros_communicator, data_processor):
#         self.ros_communicator = ros_communicator
#         self.data_processor = data_processor
#         self.finishFlag = False
#         self.global_plan_msg = None
#         self.index = 0
#         self.index_length = 0
#         self.recordFlag = 0
#         self.goal_published_flag = False

#         self.state = 'SEARCHING'
        
#         self.forward_step = 0
#         self.rotation_step = 0
#         self.rotation_wise = True  # True: clockwise, False: counterclockwise
#         self.passed_doors = 0
        
#         # 參數（可根據機器人調整）
#         self.rotation_steps = 180  # 完成一圈（360度）所需的步數
#         self.forward_steps = 200  # 每次前進的步數

#     def reset_nav_process(self):
#         self.finishFlag = False
#         self.recordFlag = 0
#         self.goal_published_flag = False

#     def finish_nav_process(self):
#         self.finishFlag = True
#         self.recordFlag = 1

#     def get_finish_flag(self):
#         return self.finishFlag


#     def camera_nav_unity(self):
#         """實現導航邏輯，返回動作字串"""
#         action_detection = self.data_processor.get_action_detection() #皮卡丘相關訊息
#         door_detection = self.data_processor.get_door_detection() #門相關訊息

#         if action_detection == None or door_detection == None:
#             if action_detection == None:
#                 print("No action detection data available.")
#             if door_detection == None:
#                 print("No door detection data available.")
#             return "STOP"
        
#         action = "STOP"
#         print(f'Current state: {self.state}, Pikachu detection: {action_detection}, Door: {door_detection}')

#         if action_detection == 'STOP':
#             return 'STOP'
#         if self.state == 'SEARCHING':
#             if self.passed_doors >= 3:
#                 # 旋轉搜尋皮卡丘
#                 self.rotation_step += 1
#                 action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
                
#                 if action_detection == 'COUNTERCLOCKWISE_ROTATION':
#                     # 發現皮卡丘在左邊
#                     self.rotation_wise = False
#                     self.rotation_step = 0
#                     action = 'COUNTERCLOCKWISE_ROTATION'
#                     print('Pikachu detected in left, turning left')
#                 elif action_detection == 'CLOCKWISE_ROTATION':
#                     # 發現皮卡丘在右邊
#                     self.rotation_wise = True
#                     self.rotation_step = 0
#                     action = 'CLOCKWISE_ROTATION'
#                     print('Pikachu detected in right, turning right')
#                 elif action_detection == 'FORWARD':
#                     # 發現皮卡丘，進入追蹤狀態
#                     self.state = 'FOLLOWING'
#                     self.rotation_step = 0
#                     action = 'FORWARD_SLOW'
#                     print('Pikachu detected, switching to FOLLOWING state')
#                 elif self.rotation_step >= self.rotation_steps:
#                     # 完成一圈旋轉
#                     self.rotation_step = 0
#                     self.state = 'MOVING_FORWARD'
#                     self.forward_step = 0
#                     action = 'FORWARD'
#                     print('No Pikachu detected after rotation, moving forward')
#             else:
#                 # 旋轉搜尋門
#                 self.rotation_step += 1
#                 action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
                
#                 if door_detection == 'COUNTERCLOCKWISE_ROTATION':
#                     # 發現門在左邊
#                     self.rotation_wise = False
#                     self.rotation_step = 0
#                     action = 'COUNTERCLOCKWISE_ROTATION'
#                     print('Door detected in left, turning left')
#                 elif door_detection == 'CLOCKWISE_ROTATION':
#                     # 發現門在右邊
#                     self.rotation_wise = True
#                     self.rotation_step = 0
#                     action = 'CLOCKWISE_ROTATION'
#                     print('Door detected in right, turning right')
#                 elif door_detection == 'FORWARD':
#                     # 發現門，進入追蹤狀態
#                     self.state = 'FOLLOWING'
#                     self.rotation_step = 0
#                     action = 'FORWARD_SLOW'
#                     print('Door detected, switching to FOLLOWING state')
#                 elif self.rotation_step >= self.rotation_steps:
#                     # 完成一圈旋轉
#                     self.rotation_step = 0
#                     self.state = 'MOVING_FORWARD'
#                     self.forward_step = 0
#                     action = 'FORWARD'
#                     print('No Door detected after rotation, moving forward')


#         elif self.state == 'FOLLOWING':
#             if self.passed_doors >= 3:
#                 # 追蹤皮卡丘
#                 if action_detection == 'STOP':
#                     action = 'STOP'
#                     print('Pikachu in middle-bottom, stopping')
#                     self.state = 'SEARCHING'
#                     self.rotation_step = 0
#                 elif action_detection == 'FORWARD':
#                     action = 'FORWARD_SLOW'
#                     print('Moving forward towards Pikachu')
#                 elif action_detection == 'COUNTERCLOCKWISE_ROTATION':
#                     # 發現皮卡丘在左邊
#                     self.rotation_wise = False
#                     action = 'COUNTERCLOCKWISE_ROTATION'
#                     print('Pikachu detected in left, turning left')
#                 elif action_detection == 'CLOCKWISE_ROTATION':
#                     # 發現皮卡丘在右邊
#                     self.rotation_wise = True
#                     action = 'CLOCKWISE_ROTATION'
#                     print('Pikachu detected in right, turning right')
#                 else:
#                     # 失去皮卡丘，回到搜尋狀態
#                     self.state = 'SEARCHING'
#                     self.rotation_step = 0
#                     action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
#                     print('Lost Pikachu, switching to SEARCHING state')
#             else:
#                 # 追蹤門
#                 if door_detection == 'FORWARD':
#                     action = 'FORWARD_SLOW'
#                     print('Moving forward towards Door')
#                 elif door_detection == 'COUNTERCLOCKWISE_ROTATION':
#                     # 發現門在左邊
#                     self.rotation_wise = False
#                     action = 'COUNTERCLOCKWISE_ROTATION'
#                     print('Door detected in left, turning left')
#                 elif door_detection == 'CLOCKWISE_ROTATION':
#                     # 發現門在右邊
#                     self.rotation_wise = True
#                     action = 'CLOCKWISE_ROTATION'
#                     print('Door detected in right, turning right')
#                 else:
#                     # 失去門，回到當通過
#                     self.state = 'SEARCHING'
#                     self.rotation_step = 0
#                     action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
#                     print('Lost Door, switching to SEARCHING state')
#                     self.passed_doors += 1

#         elif self.state == 'MOVING_FORWARD':
#             # 前進一段距離後繼續搜尋
#             self.forward_step += 1
#             action = 'FORWARD'
#             # print(f'Moving forward, step {self.forward_step}/{self.forward_steps}')
#             if self.forward_step >= self.forward_steps:
#                 self.state = 'SEARCHING'
#                 self.rotation_step = 0
#                 action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
#                 print('Finished moving forward, resuming search')

#         # 如果收到 ROTATE，重置搜尋
#         if action_detection == 'ROTATE' and (self.state == 'FOLLOWING' or self.state == 'MOVING_FORWARD'):
#             action = 'CLOCKWISE_ROTATION' if self.rotation_wise else 'COUNTERCLOCKWISE_ROTATION'
#             print('Non-Pikachu in middle-bottom, stopping')
#             self.state = 'SEARCHING'
#             self.rotation_step = 0

#         return action

#     def stop_nav(self):
#         return "STOP"

class Nav2Processing_door_random:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False

        self.state = 'SEARCHING'
        
        self.X = -8
        self.Y_step = 0
        self.rotation_angle = 0
        self.GO_Y_step = 0
        self.search_dir = True  # True: clockwise, False: counterclockwise
        self.x_to_go = 0
        self.dodof = 0
        
        # 參數（可根據機器人調整）
        self.forward_steps = 170.0  # 每次前進的步數
        self.right_rotate_angle = 90 / 46
        self.left_rotate_angle = - 90 / 52

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False

        self.rotation_angle = 0
        self.GO_Y_step = 0
        self.search_dir = True
        self.state = 'SEARCHING'
        self.x_to_go = 0
        self.X = -8
        self.Y_step = 0
        self.dodof = 0

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag


    def camera_nav_unity(self):
        """實現導航邏輯，返回動作字串"""
        action_detection = self.data_processor.get_action_detection() #皮卡丘相關訊息
        door_detection = self.data_processor.get_door_detection() #門相關訊息

        if action_detection == None or door_detection == None:
            if action_detection == None:
                print("No action detection data available.")
            if door_detection == None:
                print("No door detection data available.")
            return "STOP"
        
        action = "STOP"
        print(f'Current state: {self.state}, Door: {door_detection}')## Pikachu detection: {action_detection},

        if action_detection == 'STOP':
            return 'STOP'
        if self.state == 'SEARCHING':
            #搜尋皮卡丘
            action = 'FORWARD'
            if action_detection == 'COUNTERCLOCKWISE_ROTATION':
                # 發現皮卡丘在左邊
                self.search_dir = False
                self.rotation_angle += self.left_rotate_angle
                action = 'COUNTERCLOCKWISE_ROTATION'
                print('Pikachu detected in left, turning left')
            elif action_detection == 'CLOCKWISE_ROTATION':
                # 發現皮卡丘在右邊
                self.search_dir = True
                self.rotation_angle += self.right_rotate_angle
                action = 'CLOCKWISE_ROTATION'
                print('Pikachu detected in right, turning right')
            elif action_detection == 'FORWARD':
                # 發現皮卡丘，進入追蹤狀態
                action = 'FORWARD'
                print('Pikachu detected')
            # else:
            #     if self.search_dir:
            #         self.rotation_angle += self.right_rotate_angle
            #         action = 'CLOCKWISE_ROTATION'
            #     else:
            #         self.rotation_angle += self.left_rotate_angle
            #         action = 'COUNTERCLOCKWISE_ROTATION'
            #     print('No Pikachu detected')
            
            # else:
            # 旋轉搜尋門
            elif door_detection == 'COUNTERCLOCKWISE_ROTATION':
                # 發現門在左邊
                self.search_dir = False
                self.rotation_angle += self.left_rotate_angle
                action = 'COUNTERCLOCKWISE_ROTATION'
                print('Door detected in left, turning left')
            elif door_detection == 'CLOCKWISE_ROTATION':
                # 發現門在右邊
                self.search_dir = True
                self.rotation_angle += self.right_rotate_angle
                action = 'CLOCKWISE_ROTATION'
                print('Door detected in right, turning right')
            elif door_detection == 'FORWARD':
                # 發現門，進入追蹤狀態
                self.state = 'FACE_LEFT' if self.rotation_angle < 0 else 'FACE_RIGHT'
                action = 'STOP'
                self.dodof = 0
                print('Door detected, switching to FOLLOWING state')
            else:
                if self.search_dir:
                    if self.rotation_angle <= 90:
                        # 旋轉面向右
                        self.rotation_angle += self.right_rotate_angle
                        action = 'CLOCKWISE_ROTATION'
                    else:
                        # 旋轉面向左
                        self.search_dir = False
                        self.rotation_angle += self.left_rotate_angle
                        action = 'COUNTERCLOCKWISE_ROTATION'
                else:
                    if self.rotation_angle >= -90:
                        # 旋轉面向左
                        self.rotation_angle += self.left_rotate_angle
                        action = 'COUNTERCLOCKWISE_ROTATION'
                    else:
                        # 旋轉面向右
                        self.search_dir = True
                        self.rotation_angle += self.right_rotate_angle
                        action = 'CLOCKWISE_ROTATION'

        elif self.state == 'FACE_LEFT' or self.state == 'FACE_RIGHT':
            self.dodof += 1
            action = 'FORWARD'
            if self.dodof > 185:
                self.state = 'SEARCHING'
                action = 'STOP'
                self.GO_Y_step += 1
                
        print(f'Action: {action}, Rotation angle: {self.rotation_angle}')
        return action

    def stop_nav(self):
        return "STOP"