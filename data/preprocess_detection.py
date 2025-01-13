import os
import sys
import argparse
import h5py
from tqdm import tqdm
from dataloader_detection import computeSliceMatrix
import logging

FILE_MARKER_DIR = "D:/eeg-gnn-ssl/data/file_markers_detection"


def main(
        resample_dir,
        raw_data_dir,
        output_dir,
        clip_len,
        time_step_size,
        is_fft=True):
    with open(os.path.join(
            FILE_MARKER_DIR, "trainSet_seq2seq_" + str(clip_len) + "s_sz.txt"), "r") as f:
        train_str = f.readlines()
    train_tuples = [curr_str.strip('\n').split(',') for curr_str in train_str]
    with open(os.path.join(
            FILE_MARKER_DIR, "trainSet_seq2seq_" + str(clip_len) + "s_nosz.txt"), "r") as f:
        train_str = f.readlines()
    train_tuples = train_tuples + \
        [curr_str.strip('\n').split(',') for curr_str in train_str]

    with open(os.path.join(
            FILE_MARKER_DIR, "devSet_seq2seq_" + str(clip_len) + "s_sz.txt"), "r") as f:
        dev_str = f.readlines()
    dev_tuples = [curr_str.strip('\n').split(',') for curr_str in dev_str]
    with open(os.path.join(
            FILE_MARKER_DIR, "devSet_seq2seq_" + str(clip_len) + "s_nosz.txt"), "r") as f:
        dev_str = f.readlines()
    dev_tuples = dev_tuples + \
        [curr_str.strip('\n').split(',') for curr_str in dev_str]

    with open(os.path.join(
            FILE_MARKER_DIR, "testSet_seq2seq_" + str(clip_len) + "s_sz.txt"), "r") as f:
        test_str = f.readlines()
    test_tuples = [curr_str.strip('\n').split(',') for curr_str in test_str]
    with open(os.path.join(
            FILE_MARKER_DIR, "testSet_seq2seq_" + str(clip_len) + "s_nosz.txt"), "r") as f:
        test_str = f.readlines()
    test_tuples = test_tuples + \
        [curr_str.strip('\n').split(',') for curr_str in test_str]

    all_tuples = train_tuples + dev_tuples + test_tuples

    edf_files = []
    for path, subdirs, files in os.walk(raw_data_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    output_dir = os.path.join(
        output_dir,
        'clipLen' +
        str(clip_len) +
        '_timeStepSize' +
        str(time_step_size))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    processed_files = 0
    failed_files = []
    failed_count = 0

    delayed_results = []
    delayed_files = []
    total_files = len(all_tuples)
    pbar = tqdm(total=total_files, desc="Processing files")

    for idx, (h5_fn, _) in enumerate(all_tuples):
        edf_fn = h5_fn.split('.edf')[0] + '.edf'
        try:
            edf_fn_full = [file for file in edf_files if h5_fn.split('.edf')[0] + '.edf' in file][0]
        except IndexError:
            logging.error(f"EDF file not found for {h5_fn}")
            failed_files.append(h5_fn)
            failed_count += 1
            pbar.total -= 1
            pbar.refresh()
            continue

        clip_idx = int(h5_fn.split('_')[-1].split('.h5')[0])
        h5_fn = h5_fn.split('.edf')[0] + '.h5'

        try:
            eeg_clip, _ = computeSliceMatrix(
                h5_fn=os.path.join(resample_dir, h5_fn),
                edf_fn=edf_fn_full,
                clip_idx=clip_idx,
                time_step_size=time_step_size,
                clip_len=clip_len,
                is_fft=is_fft)

            if eeg_clip is None:
                raise ValueError("computeSliceMatrix returned None")

            output_file = os.path.join(output_dir, f"{edf_fn}_{clip_idx}.h5")
            with h5py.File(output_file, 'w') as hf:
                hf.create_dataset('clip', data=eeg_clip)

            processed_files += 1
            pbar.update(1)
            logging.debug(f"Successfully processed: {output_file}")

        except Exception as e:
            logging.error(f"Error processing file {h5_fn}: {str(e)}")
            failed_files.append(h5_fn)
            failed_count += 1
            pbar.total -= 1
            pbar.refresh()

        if (idx + 1) % 1000 == 0:
            logging.info(f"Progress: Processed {processed_files}, Failed {failed_count}, Total {idx + 1}")

    pbar.close()

    # 최종 결과 출력
    logging.info("=" * 50)
    logging.info("Processing completed!")
    logging.info(f"Processed files: {processed_files}")
    logging.info(f"Failed files: {failed_count}")
    logging.info(f"Total files attempted: {total_files}")
    logging.info(f"Success rate: {processed_files / total_files * 100:.2f}%")
    
    if failed_files:
        logging.info("First 10 failed files:")
        for file in failed_files[:10]:
            logging.info(file)
        if len(failed_files) > 10:
            logging.info(f"... and {len(failed_files) - 10} more")
    
    logging.info("=" * 50)
    print("\nPreprocessing DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resampled_dir",
        type=str,
        default=None,
        help="Directory to resampled signals.")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Directory to raw edf files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory.")
    parser.add_argument(
        "--clip_len",
        type=int,
        default=60,
        help="EEG clip length in seconds.")
    parser.add_argument(
        "--time_step_size",
        type=int,
        default=1,
        help="Time step size in seconds.")
    parser.add_argument(
        "--is_fft",
        action="store_true",
        default=False,
        help="Whether to perform FFT.")

    args = parser.parse_args()
    main(
        args.resampled_dir,
        args.raw_data_dir,
        args.output_dir,
        args.clip_len,
        args.time_step_size,
        args.is_fft)