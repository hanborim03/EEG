import sys
import logging
import os
import numpy as np
import pyedflib
import h5py
import scipy
from tqdm import tqdm
import argparse

sys.path.append("D:/eeg-gnn-ssl")
from constants import INCLUDED_CHANNELS, FREQUENCY
from data_utils import resampleData, getEDFsignals, getOrderedChannels

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resample_all(raw_edf_dir, save_dir):
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if name.endswith(".edf"):
                edf_files.append(os.path.join(path, name))

    logging.info(f"Total EDF files found: {len(edf_files)}")

    failed_files = []
    processed_files = 0

    for edf_fn in tqdm(edf_files, desc="Processing EDF files"):
        save_fn = os.path.join(save_dir, os.path.splitext(os.path.basename(edf_fn))[0] + ".h5")
        
        if os.path.exists(save_fn):
            logging.info(f"File already exists, skipping: {save_fn}")
            processed_files += 1
            continue

        try:
            with pyedflib.EdfReader(edf_fn) as f:
                orderedChannels = getOrderedChannels(
                    edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
                )
                signals = getEDFsignals(f)
                signal_array = np.array(signals[orderedChannels, :])
                sample_freq = f.getSampleFrequency(0)

            if sample_freq != FREQUENCY:
                signal_array = resampleData(
                    signal_array,
                    to_freq=FREQUENCY,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
                hf.create_dataset("resample_freq", data=FREQUENCY)

            if os.path.exists(save_fn):
                logging.info(f"Successfully saved: {save_fn}")
                processed_files += 1
            else:
                logging.error(f"Failed to save file: {save_fn}")
                failed_files.append(edf_fn)

        except Exception as e:
            logging.error(f"Error processing file {edf_fn}: {str(e)}")
            failed_files.append(edf_fn)

    logging.info(f"Processing completed. Processed files: {processed_files}, Failed files: {len(failed_files)}")
    if failed_files:
        logging.info("Failed files:")
        for file in failed_files:
            logging.info(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resample.")
    parser.add_argument(
        "--raw_edf_dir",
        type=str,
        required=True,
        help="Full path to raw edf files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Full path to dir to save resampled signals.",
    )
    args = parser.parse_args()

    # 저장 디렉토리 확인 및 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        logging.info(f"Created save directory: {args.save_dir}")

    resample_all(args.raw_edf_dir, args.save_dir)