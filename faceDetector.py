import cv2
import mediapipe as mp

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
