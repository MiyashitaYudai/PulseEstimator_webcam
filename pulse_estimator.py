import cv2
import sys
import math
import time
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


class personFaceParts:
    def __init__(self, face_detection_result, width, height):
        self.mp_face_detection = mp.solutions.face_detection

        # 顔のキーポイントを取得
        right_eye_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        left_eye_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.LEFT_EYE)
        nose_point_norm = self.mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.NOSE_TIP)
        right_ear_point_norm = mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
        left_ear_point_norm = mp_face_detection.get_key_point(face_detection_result, self.mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)

        self.right_eye_point = self.convert_image_size_point(right_eye_point_norm, width, height)
        self.left_eye_point = self.convert_image_size_point(left_eye_point_norm, width, height)
        self.nose_point = self.convert_image_size_point(nose_point_norm, width, height)
        self.right_ear_point = self.convert_image_size_point(right_ear_point_norm, width, height)
        self.left_ear_point = self.convert_image_size_point(left_ear_point_norm, width, height)


    def convert_image_size_point(self, point, width, height):
        '''
        座標を画像座標に変換
        '''
        return (int(point.x * width), int(point.y * height))

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



class faceDetector(object):

    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.person_face_list = []
    

    def detect(self, image):
        height, width, _ = image.shape[:3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)

        if not results.detections:
            return

        # 検出結果の取得
        self.person_face_list.clear()
        for detection in results.detections:
            person = personFaceParts(detection, width, height)
            self.person_face_list.append(person)


        return self.person_face_list

class pulseEstimator:
    def __init__(self):
        self.mean_value_data_buffer = []
        self.buffer_size = 250
        self.start_time = time.time()
        self.times = []
        return

    def set_face_detect_results(self, face_detect_results):
        self.face_detect_results = face_detect_results

    def pulse_estimate(self, image):
        self.times.append(time.time() - self.start_time)

        bpm = 0
        for person in self.face_detect_results:
            forehead_rect = person.get_forehead_rect()
            mean_value = person.get_subface_means(image, forehead_rect)
            self.mean_value_data_buffer.append(mean_value)

            # 指定したサイズだけ輝度値を保持する
            self.data_length = len(self.mean_value_data_buffer)
            if self.data_length > self.buffer_size:
                self.mean_value_data_buffer = self.mean_value_data_buffer[-self.buffer_size:]
                self.times = self.times[-self.buffer_size:]
                self.data_length = self.buffer_size

            bpm = self.process()
            break

        text = "(estimate: %0.1f bpm)" % (bpm)
        cv2.putText(image, text, (int(200), int(200)), cv2.FONT_HERSHEY_PLAIN, 4, 2)
        cv2.imshow('MediaPipe Face Detection', image)
        return bpm

    def process(self):
        processed = np.array(self.mean_value_data_buffer)
        if not self.data_length > 10:
            return 0

        fps = float(self.data_length) / (self.times[-1] - self.times[0])

        # 各輝度値データを取得した時間を、均等に分割した時間で割り当てる
        even_times = np.linspace(self.times[0], self.times[-1], self.data_length)

        # 線形補完
        interpolated = np.interp(even_times, self.times, processed)

        # ハミング窓をかける
        interpolated = np.hamming(self.data_length) * interpolated

        # 平均値からの差分を取得
        interpolated = interpolated - np.mean(interpolated)

        # フーリエ変換
        raw = np.fft.rfft(interpolated)

        # 周期を取得
        period = np.abs(raw)

        # 周波数
        freqs = float(fps) / self.data_length * np.arange(self.data_length / 2 + 1)
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



def plot_BPM(bpm_data, plot_image, size):

    n_plots = len(bpm_data)
    if n_plots < 2:
        return plot_image
     
    if n_plots >= size[0]:
        plot_image = cv2.line(plot_image, (0, 0), (1, size[1]), (0, 0, 0), 1)
        plot_image = np.roll(plot_image, -1, axis=1)

    idx = n_plots - 2
    bpm_start = float(bpm_data[idx])
    bpm_end = float(bpm_data[idx + 1])

    bpm_start_norm = int(size[1] - ((bpm_start / 150.0) * size[1]))
    bpm_end_norm = int(size[1] - ((bpm_end / 150.0) * size[1]))

    plot_start = (n_plots - 2, bpm_start_norm)
    plot_end = (n_plots - 1, bpm_end_norm)

    cv2.line(plot_image, plot_start, plot_end, (255,255,255),1)    
    cv2.imshow("BPM", plot_image)

    return plot_image



if __name__ == "__main__":
    # カメラ接続
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        sys.exit()

    face_detector = faceDetector()
    pulse_estimator = pulseEstimator()

    bpm_list = []
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
        pulse_estimator.set_face_detect_results(face_detect_results)

        # 脈拍の計測
        bpm = pulse_estimator.pulse_estimate(image)
        bpm_list.append(bpm)

        bpm_list = bpm_list[-640:]
        plot_image = plot_BPM(bpm_list, plot_image, size)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
