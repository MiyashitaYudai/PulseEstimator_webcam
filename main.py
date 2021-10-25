import cv2
import sys
import numpy as np

from userTracker import userTracker
from faceDetector import faceDetector
from pulseEstimator import pulseEstimator

COLOR = [
    (100, 100, 200), (100, 200, 100), (200, 100, 100),
    (0, 0, 200), (0, 200, 0), (200, 0, 0)
    ]

def plot_BPM(tracker, plot_image, size):

    max_plots_length = 0
    for person_idx, person in enumerate(tracker.tracking_person_list):
        if max_plots_length < len(person.bpm_list):
            max_plots_length = len(person.bpm_list)

    if max_plots_length >= size[0]:
        plot_image = cv2.line(plot_image, (0, 0), (1, size[1]), (0, 0, 0), 1)
        plot_image = np.roll(plot_image, -1, axis=1)



    for person_idx, person in enumerate(tracker.tracking_person_list):
        n_plots = len(person.bpm_list)
        if n_plots < 2:
            continue
        
        idx = n_plots - 2
        bpm_start = float(person.bpm_list[idx])
        bpm_end = float(person.bpm_list[idx + 1])

        bpm_start_norm = int(size[1] - ((bpm_start / 150.0) * size[1]))
        bpm_end_norm = int(size[1] - ((bpm_end / 150.0) * size[1]))

        plot_start = (max_plots_length - 2, bpm_start_norm)
        plot_end = (max_plots_length - 1, bpm_end_norm)

        cv2.line(plot_image, plot_start, plot_end, COLOR[person_idx], 1)
        text = "(estimate: %0.1f bpm)" % (person.bpm_list[-1])
        cv2.rectangle(plot_image, (size[0] - 130, person_idx * 15), (size[0], (person_idx + 1) * 15), (0, 0, 0), -1)
        cv2.putText(plot_image, text, (size[0] - 130, (person_idx + 1) * 14), cv2.FONT_HERSHEY_PLAIN, 0.6, COLOR[person_idx])

    return plot_image

def plot_value(tracker, plot_image, size):

    for person in tracker.tracking_person_list:
        n_plots = len(person.mean_value_data_list)
        if n_plots < 2:
            return plot_image
        
        if n_plots >= 250:
            plot_image = cv2.line(plot_image, (0, 0), (1, size[1]), (0, 0, 0), 1)
            plot_image = np.roll(plot_image, -1, axis=1)

        idx = n_plots - 2
        bpm_start = float(person.mean_value_data_list[idx])
        bpm_end = float(person.mean_value_data_list[idx + 1])

        bpm_start_norm = int(size[1] - ((bpm_start / 150.0) * size[1]))
        bpm_end_norm = int(size[1] - ((bpm_end / 150.0) * size[1]))

        plot_start = (n_plots - 2, bpm_start_norm)
        plot_end = (n_plots - 1, bpm_end_norm)

        cv2.line(plot_image, plot_start, plot_end, (255,255,255),1)    

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
    plot_image = np.zeros((size[1], size[0], 3), np.uint8)

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

        # 顔トラッキングの結果描画
        for person in tracker.tracking_person_list:
            image = person.draw_face_keypoints(image)

        cv2.imshow("camera image", image)
        cv2.imshow("BPM", plot_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
