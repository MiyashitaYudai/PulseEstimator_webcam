import enum
import cv2
import sys
import math
import time
import numpy as np
import mediapipe as mp
from scipy.optimize import linear_sum_assignment


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

class userTracker:
    def __init__(self, width, height):
        self.new_id = 0
        self.width = width
        self.height = height
        self.delete_time_sec = 3.0
        self.tracking_person_list = []

    def update(self, detection_result):
        # トラッキングしている人物がいない場合はそのまま追跡対象に加える
        if len(self.tracking_person_list) == 0:
            for detection in detection_result:
                person = personFaceParts(detection, self.width, self.height, self.new_id)
                self.tracking_person_list.append(person)
                self.new_id += 1
            return

        # トラッキングするためにコストのマトリクスを生成する
        new_person_list = []
        cost_matrix = np.zeros((len(detection_result), len(self.tracking_person_list)))
        for idx, detection in enumerate(detection_result):
            person = personFaceParts(detection, self.width, self.height, -1)
            new_person_list.append(person)
            person_mean_pos = person.get_mean_pos()

            for tracking_idx, tracking_person in enumerate(self.tracking_person_list):
                tracking_person_mean_pos = tracking_person.get_mean_pos()
                diff = person_mean_pos - tracking_person_mean_pos
                distance = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
                cost_matrix[idx][tracking_idx] = distance

        
        # ハンガリアン・アルゴリズムによる最適割り当て問題
        detected_idx, tracking_idx = linear_sum_assignment(cost_matrix)

        # 座標情報の更新
        for idx, person in enumerate(self.tracking_person_list):
            match_idx =  np.where(tracking_idx == idx)[0]

            # 追跡できなかった場合
            if len(match_idx) == 0:
                person.track_failed()
                continue

            # 座標の更新
            person.update(new_person_list[match_idx[0]])

        # 新規で追跡対象を追加
        for idx, new_person in enumerate(new_person_list):
            match_idx =  np.where(detected_idx == idx)[0]
            if len(match_idx) == 0:
                new_person.id = self.new_id
                self.tracking_person_list.append(new_person)
                self.new_id += 1
                continue

        # 一定時間トラッキングされないユーザーを削除する
        for idx, person in enumerate(self.tracking_person_list):
            total_tracking_time = time.time() - person.track_start_time
            if total_tracking_time - person.time_log[-1] > self.delete_time_sec:
                self.tracking_person_list.pop(idx)
                 
class faceDetector(object):

    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)

        if not results.detections:
            return None

        return results.detections


class pulseEstimator:
    def __init__(self):
        self.buffer_size = 250
        return

    def pulse_estimate(self, image, tracker):
        bpm = 0
        for person in tracker.tracking_person_list:
            if person.is_track_success:
                forehead_rect = person.get_forehead_rect()
                mean_value = person.get_subface_means(image, forehead_rect)
                person.mean_value_data_list.append(mean_value)

            # 指定したサイズだけ輝度値を保持する
            data_length = len(person.mean_value_data_list)
            if data_length > self.buffer_size:
                person.mean_value_data_list = person.mean_value_data_list[-self.buffer_size:]
                person.time_log = person.time_log[-self.buffer_size:]
                data_length = self.buffer_size

            bpm = self.process(person)
            person.bpm_list.append(bpm)


    def process(self, person):
        data_length = len(person.mean_value_data_list)
        processed = np.array(person.mean_value_data_list)
        if not data_length > 10:
            return 0

        fps = float(data_length) / (person.time_log[-1] - person.time_log[0])

        # 各輝度値データを取得した時間を、均等に分割した時間で割り当てる
        even_times = np.linspace(person.time_log[0], person.time_log[-1], data_length)

        # 線形補完
        interpolated = np.interp(even_times, person.time_log, processed)

        # ハミング窓をかける
        interpolated = np.hamming(data_length) * interpolated

        # 平均値からの差分を取得
        interpolated = interpolated - np.mean(interpolated)

        # フーリエ変換
        raw = np.fft.rfft(interpolated)

        # 周期を取得
        period = np.abs(raw)

        # 周波数
        freqs = float(fps) / data_length * np.arange(data_length / 2 + 1)
        freqs = 60. * freqs

        # 50~180以内のデータをピック
        idx = np.where((freqs > 50) & (freqs < 180))
        if len(idx[0]) == 0:
            return 0

        period = period[idx]
        freqs = freqs[idx]

        # 周期のピークを取得
        peak = np.argmax(period)
        bpm = freqs[peak]

        return bpm



def plot_BPM(tracker, plot_image, size):

    for person in tracker.tracking_person_list:
        n_plots = len(person.bpm_list)
        if n_plots < 2:
            return plot_image
        
        if n_plots >= size[0]:
            plot_image = cv2.line(plot_image, (0, 0), (1, size[1]), (0, 0, 0), 1)
            plot_image = np.roll(plot_image, -1, axis=1)

        idx = n_plots - 2
        bpm_start = float(person.bpm_list[idx])
        bpm_end = float(person.bpm_list[idx + 1])

        bpm_start_norm = int(size[1] - ((bpm_start / 150.0) * size[1]))
        bpm_end_norm = int(size[1] - ((bpm_end / 150.0) * size[1]))

        plot_start = (n_plots - 2, bpm_start_norm)
        plot_end = (n_plots - 1, bpm_end_norm)

        cv2.line(plot_image, plot_start, plot_end, (255,255,255),1)    
        text = "(estimate: %0.1f bpm)" % (person.bpm_list[-1])
        cv2.rectangle(plot_image, (size[0] - 200, 0), (size[0], 80), (0, 0, 0), -1)
        cv2.putText(plot_image, text, (size[0] - 200, 30), cv2.FONT_HERSHEY_PLAIN, 1, 2)

    return plot_image



if __name__ == "__main__":
    # カメラ接続
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        sys.exit()


    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    face_detector = faceDetector()
    pulse_estimator = pulseEstimator()
    tracker = userTracker(width, height)

    size = (640, 280)
    plot_image = np.zeros((size[1], size[0]))

    while camera.isOpened():
        # カメラ画像取得
        ret, image = camera.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # 顔の検出
        face_detect_results = face_detector.detect(image)
        if face_detect_results is None:
            continue

        # トラッキング        
        tracker.update(face_detect_results)

        # 脈拍の計測
        pulse_estimator.pulse_estimate(image, tracker)

        # 脈拍のプロット
        for person in tracker.tracking_person_list:
            if len(person.bpm_list) >= size[0]:
                person.bpm_list = person.bpm_list[-size[0]:]

        plot_image = plot_BPM(tracker, plot_image, size)    
        cv2.imshow("BPM", plot_image)

        # 顔トラッキングの結果描画
        for person in tracker.tracking_person_list:
            image = person.draw_face_keypoints(image)

        cv2.imshow("camera image", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
