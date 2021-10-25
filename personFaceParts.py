import cv2
import math
import time
import numpy as np
import mediapipe as mp

from oneEuroFilter import OneEuroFilter


class personFaceParts:
    def __init__(self, face_detection_result, width, height, id):
        self.mp_face_detection = mp.solutions.face_detection
        self.width = width
        self.height = height
        self.id = id
        self.is_track_success = True
        self.bpm_list = []
        self.mean_value_data_list = []

        config = {
            'freq': 120,
            'mincutoff': 0.5,
            'beta': 0.007,
            'dcutoff': 1.0
        }

        self.filter = OneEuroFilter(**config)

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


    def get_search_rects(self):
        '''
        探索する矩形を取得
        '''
        scale = math.sqrt(math.pow(self.right_eye_point[0] - self.left_eye_point[0], 2) + math.pow(self.right_eye_point[1] - self.left_eye_point[1], 2))
        scale = int(scale / 3)


        rects = []
        right_cheek_rect = (int(self.right_eye_point[0] - scale), int(self.nose_point[1]), scale, scale)
        left_cheek_rect = (int(self.left_eye_point[0]), int(self.nose_point[1]), scale, scale)

        rects.append(right_cheek_rect)
        rects.append(left_cheek_rect)

        # mean_eye_point = ((self.right_eye_point[0] + self.left_eye_point[0]) / 2.0, (self.right_eye_point[1] + self.left_eye_point[1]) / 2.0)
        # forehead_point = (int(2 * mean_eye_point[0] - self.nose_point[0]), int(2 * mean_eye_point[1] - self.nose_point[1]))
        # forehead_rect = (int(forehead_point[0] - scale / 2), int(forehead_point[1] - scale / 2), scale, scale)

        return rects

    def draw_face_keypoints(self, image):
        '''
        パーツの描画
        '''
        cv2.circle(image, self.right_eye_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.left_eye_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.nose_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.right_ear_point, 3, (100, 100, 200), -1)
        cv2.circle(image, self.left_ear_point, 3, (100, 100, 200), -1)
        
        rects = self.get_search_rects()
        for rect in rects:
            forehead_point = np.array([int((rect[0] + rect[2] / 2)), int((rect[1] + rect[3] / 2))])
            cv2.rectangle(
                image, 
                (rect[0], rect[1]),
                (rect[0] + rect[2], rect[1] + rect[3]),
                (100, 100, 200), 2)

        text = "user ID : " + str(self.id)
        cv2.putText(image, text, forehead_point, cv2.FONT_HERSHEY_PLAIN, 1, 2)

        return image

    def get_subface_means(self, image):
        '''
        矩形内の平均輝度を取得
        '''

        blur_image = cv2.medianBlur(image, 5)

        rects = self.get_search_rects()
        value_sum = 0
        for rect in rects:
            x, y, w, h = rect
            subframe = blur_image[y:y + h, x:x + w, :]
            v1 = np.mean(subframe[:, :, 0])         # B
            v2 = np.mean(subframe[:, :, 1])         # G
            v3 = np.mean(subframe[:, :, 2])         # R
            value_sum +=(v1 * 0.1 + v2 * 0.8 + v3 * 0.1)

        value_mean = value_sum / len(rects)

        return value_mean

    def get_mean_pos(self):
        sum_pos = (self.right_eye_point + self.left_eye_point + self.nose_point + self.right_ear_point + self.left_ear_point)
        mean_pos = np.array([int(sum_pos[0] / 5.0), int(sum_pos[1] / 5.0)])
        return mean_pos

    def add_bpm(self, bpm):
        filtered_bpm = self.filter(bpm)
        self.bpm_list.append(filtered_bpm)
