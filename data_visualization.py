import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화
print("GPU 비활성화 - CPU만 사용")

import mne
import matplotlib.pyplot as plt

# EEG 데이터 로드 및 시각화
file_path = 'D:/eeg-gnn-ssl/data/v2.0.3/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# 시각화
fig = raw.plot(n_channels=15, duration=10, scalings='auto', color='blue')

# 모든 선의 굵기를 2로 설정
for ax in fig.axes:
    for line in ax.lines:
        line.set_linewidth(2)

plt.show()