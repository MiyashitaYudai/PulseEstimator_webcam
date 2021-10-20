import cv2
import math
import time
import numpy as np
import mediapipe as mp


class personFaceParts:
    def __init__(self, face_detection_result, width, height, id):
        self.mp_face_detection = mp.solutions.face_detection
        self.width = width
        self.height = height
        self.id = id
        self.is_track_success = True
        self.bpm_list = []
        self.mean_value_data_list = []

        # 追跡時間
        self.track_start_time = time.time()
        self.time_log = []
        self.time_log.append(0.0)

        # 顔のキーポイントを取得
        right_eye_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        left_eye_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.LEFT_EYE)
        nose_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.NOSE_TIP)
        right_ear_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
        left_ear_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)

        # 画像座標に変換
        self.right_eye_point = self.convert_image_size_point(right_eye_point_norm)
        self.left_eye_point = self.convert_image_size_point(left_eye_point_norm)
        self.nose_point = self.convert_image_size_point(nose_point_norm)
        self.right_ear_point = self.convert_image_size_point(right_ear_point_norm)
        self.left_ear_point = self.convert_image_size_point(left_ear_point_norm)


    def update(self, person_face_parts):
        '''
        ユーザーの情報を更新
        '''
        self.is_track_success = True
        self.time_log.append(time.time() - self.track_start_time)

        self.right_eye_point = person_face_parts.right_eye_point
        self.left_eye_point = person_face_parts.left_eye_point
        self.nose_point = person_face_parts.nose_point
        self.right_ear_point = person_face_parts.right_ear_point
        self.left_ear_point = person_face_parts.left_ear_point

    def track_failed(self):
        self.is_track_success = False        

    def convert_image_size_point(self, point):
        '''
        座標を画像座標に変換
        '''
        return np.array([int(point.x * self.width), int(point.y * self.height)])

    def get_forehead_rect(self):
        '''
        おでこの矩形を取得
        '''
        scale = math.sqrt(math.pow(self.right_eye_point[0] - self.left_eye_point[0], 2) + math.pow(self.right_eye_point[1] - self.left_eye_point[1], 2))
        scale = int(scale / 2)

        mean_eye_point = ((self.right_eye_point[0] + self.left_eye_point[0]) / 2.0, (self.right_eye_point[1] + self.left_eye_point[1]) / 2.0)

        forehead_point = (int(2 * mean_eye_point[0] - self.nose_point[0]), int(2 * mean_eye_point[1] - self.nose_point[1]))
        forehead_rect = (int(forehead_point[0] - scale / 2), int(forehead_point[1] - scale / 2), scale, scale)

        return forehead_rect

    def draw_face_keypoints(self, image):
        '''
        パーツの描画
        '''
        cv2.circle(image, self.right_eye_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.left_eye_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.nose_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.right_ear_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.left_ear_point, 3, (100, 100, 200), -1)

        forehead_rect = self.get_forehead_rect()
        forehead_point = np.array([int((forehead_rect[0] + forehead_rect[2] / 2)), int((forehead_rect[1] + forehead_rect[3] / 2))])

        text = "user ID : " + str(self.id)
        cv2.putText(image, text, forehead_point, cv2.FONT_HERSHEY_PLAIN, 1, 2)

        return image

    def get_subface_means(self, image, rect):
        '''
        矩形内の平均輝度を取得
        '''
        x, y, w, h = rect
        subframe = image[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])         # B
        v2 = np.mean(subframe[:, :, 1])         # G
        v3 = np.mean(subframe[:, :, 2])         # R

        return (v1 + v2 + v3) / 3.

    def get_mean_pos(self):
        sum_pos = (self.right_eye_point + self.left_eye_point + self.nose_point + self.right_ear_point + self.left_ear_point)
        mean_pos = np.array([int(sum_pos[0] / 5.0), int(sum_pos[1] / 5.0)])
        return mean_pos
