import os
import csv
import logging
import random

# 발작이 있는 라벨 목록
seizure_labels = {'seiz', 'fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'atsz', 'mysz', 'nesz'}

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EDF 파일 가져오기 함수
def get_edf_files(directory):
    edf_files = []
    for root, _, files in os.walk(directory):
        # logging.info(f"Checking directory: {root}")  # 디버깅용 출력
        for file in files:
            if file.lower().endswith(".edf"):  # 대소문자 구분 없이 .edf 파일 찾기
                edf_files.append(os.path.join(root, file))
    logging.info(f"Total EDF files found: {len(edf_files)}")
    return edf_files

# CSV_BI 파일에서 발작 정보를 읽고 클립 인덱스를 생성하는 함수
def process_csv_file(patient_id, session_num, token_num, csv_bi_file_path):
    clips = []
    clip_index = 0
    window_size = 12.0
    
    # CSV_BI 파일이 존재하는지 확인
    if not os.path.exists(csv_bi_file_path):
        logging.error(f"CSV_BI file not found: {csv_bi_file_path}")
        return clips
    
    logging.info(f"Processing CSV_BI file: {csv_bi_file_path}")
    
    with open(csv_bi_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # 주석(#)으로 시작하는 줄을 건너뛰기
        for row in reader:
            if row[0].startswith('#'):
                continue  # 주석 줄은 건너뜁니다.
            
            # 첫 번째 데이터 줄로부터 헤더를 설정하고 DictReader로 변환
            fieldnames = ['channel', 'start_time', 'stop_time', 'label', 'confidence']
            break  # 첫 번째 데이터 줄을 찾았으므로 루프를 종료합니다.
        
        # DictReader로 데이터를 다시 읽기 (주석 제외)
        #csvfile.seek(0)  # 파일 포인터를 처음으로 되돌립니다.
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        
        
        for row in reader:
            if row['start_time'] == 'start_time':  # 헤더 행은 건너뛰기
                continue
            start_time = float(row['start_time'])
            stop_time = float(row['stop_time'])
            label = row['label']
            print('시작, 끝', start_time, stop_time, label)
            
            # 발작 여부에 따라 라벨 설정 (발작이 있으면 1, 없으면 0)
            seizure_label = 1 if label in seizure_labels else 0
            
            # 클립 인덱스 계산
            current_time = start_time
            while current_time + window_size <= stop_time:
                clip_index += 1
                current_time = current_time+window_size
                print('현재시간', current_time)
            
                # 새로운 파일 이름 생성 (파일 경로가 아닌 순수 파일 이름만 생성)
                new_file_name = f"{patient_id}_s{session_num:03d}_t{token_num:03d}.edf_{clip_index}.h5,{seizure_label}"
            
            
                # 클립 정보 저장
                clips.append(new_file_name)
    
    return clips

# 텍스트 파일로 저장하는 함수
def save_to_txt(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")

# 전체 데이터셋 처리 함수
def process_dataset(root_dir):
    # 각 디렉토리 내의 EDF 파일 처리
    edf_files = get_edf_files(root_dir)
    
    test_sz, test_nosz = [], []
    train_sz, train_nosz = [], []
    validation_sz, validation_nosz = [], []

    for edf_file in edf_files:
        # 환자 ID와 세션/토큰 번호 추출 (예시: aaaaajy_s001_t000.edf)
        base_name = os.path.basename(edf_file).replace('.edf', '')
        patient_id, session_num_token = base_name.split('_s')
        session_num, token_num = map(int, session_num_token.split('_t'))
        
        # CSV_BI 파일 경로 생성 (EDF와 동일한 이름의 CSV_BI가 있다고 가정)
        csv_bi_file_path = edf_file.replace('.edf', '.csv_bi')
        
        if not os.path.exists(csv_bi_file_path):
            logging.error(f"CSV_BI file not found for {edf_file}")
            continue
        
        logging.info(f"Processing EDF file: {edf_file}")
        
        # CSV_BI 파일 처리하여 12초 단위 클립 리스트 생성
        clips = process_csv_file(patient_id, session_num, token_num, csv_bi_file_path)
        
        if not clips:  # 클립 리스트가 비어 있는 경우 처리 중단
            logging.warning(f"No data processed from {csv_bi_file_path}")
            continue
        
        # dev 디렉토리는 테스트 데이터셋으로 사용
        if 'dev' in edf_file:
            for clip in clips:
                if clip.endswith(',1'):  # 발작 있음
                    test_sz.append(clip)
                else:  # 발작 없음
                    test_nosz.append(clip)
        
        # train 디렉토리는 훈련 및 검증 데이터셋으로 사용 (9:1 비율로 나누기)
        elif 'train' in edf_file:
            random.shuffle(clips)
            split_idx = int(len(clips) * 0.9) if len(clips) > 0 else None
            
            if split_idx is not None:
                train_clips, validation_clips = clips[:split_idx], clips[split_idx:]
                
                for clip in train_clips:
                    if clip.endswith(',1'):  # 발작 있음
                        train_sz.append(clip)
                    else:  # 발작 없음
                        train_nosz.append(clip)
                
                for clip in validation_clips:
                    if clip.endswith(',1'):  # 발작 있음
                        validation_sz.append(clip)
                    else:  # 발작 없음
                        validation_nosz.append(clip)

    # 결과를 텍스트 파일로 저장
    save_to_txt(test_sz, 'testSet_sz.txt')
    save_to_txt(test_nosz, 'testSet_nosz.txt')
    save_to_txt(train_sz, 'trainSet_sz.txt')
    save_to_txt(train_nosz, 'trainSet_nosz.txt')
    save_to_txt(validation_sz, 'validationSet_sz.txt')
    save_to_txt(validation_nosz, 'validationSet_nosz.txt')

# 루트 디렉토리 지정 및 실행
root_dir = os.path.join("D:\\", "eeg-gnn-ssl", "data", "v2.0.3", "edf")
print(root_dir)
process_dataset(root_dir)

logging.info("파일 저장이 완료되었습니다.")