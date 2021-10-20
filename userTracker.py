import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from personFaceParts import personFaceParts

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
                 
