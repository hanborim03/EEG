import torch
import numpy as np

# 1. 가우시안 노이즈 추가
def add_gaussian_noise(eeg_data, noise_level=0.01):
    """
    EEG 데이터에 가우시안 노이즈를 추가하는 함수
    :param eeg_data: 원본 EEG 데이터 (배치 크기 x 채널 수 x 시간 길이)
    :param noise_level: 노이즈 강도 (표준편차)
    :return: 노이즈가 추가된 EEG 데이터
    """
    noise = torch.randn_like(eeg_data) * noise_level
    return eeg_data + noise

# 2. 채널 드롭아웃
def channel_dropout(eeg_data, dropout_prob=0.1):
    """
    일부 채널에 드롭아웃을 적용하는 함수
    :param eeg_data: 원본 EEG 데이터 (배치 크기 x 채널 수 x 시간 길이)
    :param dropout_prob: 드롭아웃 확률
    :return: 드롭아웃이 적용된 EEG 데이터
    """
    mask = torch.rand(eeg_data.shape[0], eeg_data.shape[1], 1) > dropout_prob
    return eeg_data * mask

# 3. 시간 축 왜곡 (Time Stretch)
def time_stretch(eeg_data, stretch_factor=1.2):
    """
    EEG 신호의 시간 축을 늘리거나 줄이는 함수 (시간 왜곡)
    :param eeg_data: 원본 EEG 데이터 (배치 크기 x 채널 수 x 시간 길이)
    :param stretch_factor: 시간 축 왜곡 비율 (1보다 크면 늘어남, 작으면 줄어듦)
    :return: 시간 축이 왜곡된 EEG 데이터
    """
    # 시간 축을 늘리거나 줄이는 방법으로 scipy의 resample 함수를 사용할 수 있습니다.
    from scipy.signal import resample
    
    batch_size, num_channels, time_length = eeg_data.shape
    new_time_length = int(time_length * stretch_factor)
    
    stretched_eeg = np.zeros((batch_size, num_channels, new_time_length))
    
    for i in range(batch_size):
        for j in range(num_channels):
            stretched_eeg[i, j, :] = resample(eeg_data[i, j, :], new_time_length)
    
    return torch.tensor(stretched_eeg)

# 4. 신호 증폭/감쇠 (Amplitude Scaling)
def amplitude_scaling(eeg_data, scale_factor=1.5):
    """
    신호의 진폭을 증가시키거나 감소시키는 함수
    :param eeg_data: 원본 EEG 데이터 (배치 크기 x 채널 수 x 시간 길이)
    :param scale_factor: 진폭 스케일링 비율 (1보다 크면 증폭, 작으면 감쇠)
    :return: 진폭이 조정된 EEG 데이터
    """
    return eeg_data * scale_factor

# 5. 시간 이동 (Time Shift)
def time_shift(eeg_data, shift_max=0.2):
    """
    신호를 일정 시간만큼 앞으로 또는 뒤로 이동시키는 함수
    :param eeg_data: 원본 EEG 데이터 (배치 크기 x 채널 수 x 시간 길이)
    :param shift_max: 최대 이동 비율 (전체 길이에 대한 비율)
    :return: 시간이 이동된 EEG 데이터
    """
    batch_size, num_channels, time_length = eeg_data.shape
    shift = int(np.random.uniform(-shift_max, shift_max) * time_length)

    if shift > 0:
        shifted_eeg = torch.cat([eeg_data[:, :, shift:], torch.zeros(batch_size, num_channels, shift)], dim=2)
    else:
        shifted_eeg = torch.cat([torch.zeros(batch_size, num_channels, -shift), eeg_data[:, :, :shift]], dim=2)

    return shifted_eeg

# 예시 사용법:
eeg_data = torch.randn(32, 64, 1000)  # 배치 크기 32, 채널 64개, 시간 길이 1000인 임의의 EEG 데이터

# 가우시안 노이즈 추가
eeg_with_noise = add_gaussian_noise(eeg_data)

# 채널 드롭아웃 적용
eeg_with_dropout = channel_dropout(eeg_data)

# 시간 축 왜곡 적용
eeg_stretched = time_stretch(eeg_data)

# 신호 증폭/감쇠 적용
eeg_scaled = amplitude_scaling(eeg_data)

# 시간 이동 적용
eeg_shifted = time_shift(eeg_data)