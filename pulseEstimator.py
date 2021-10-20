import numpy as np

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
