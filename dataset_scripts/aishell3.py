import os
import re
import argparse

from tqdm import tqdm


def get_audiopath_textnorm(filepath, filename):
    with open(filepath + "/" + filename, "r") as f:
        filetext = f.read()
    lines = filetext.split("\n")

    sid = -1
    readers = {}
    for l in lines[5:]:
        if len(l) > 1:
            try:
                audio_path, pinyin, text = l.split("|")
                reader_id = audio_path[:7]
                if reader_id not in readers:
                    sid += 1
                    readers.update({reader_id: str(sid)})

                audio_path = filepath + "/wav/" + reader_id + "/" + audio_path + ".wav"
                audiopath_and_text = "|".join(
                    [audio_path, readers[reader_id], text]) + "\n"
                yield audiopath_and_text
            except:
                print(f'Skipping line: {l}, length: {len(l)}.')


def main(dataset_path, output_dir):
    # Obtain the dataset directories and collect tsv files
    print(f"Processing on {dataset_path}.")

    if dataset_path.split("/")[-1] == "train":
        filename = "label_train-set.txt"
        if os.path.exists(dataset_path + "/" + filename):
            with open(output_dir, "w", encoding="utf-8") as f:
                f.writelines(get_audiopath_textnorm(
                    dataset_path, filename="label_train-set.txt"))
        else:
            print(f"Could not find {filename}!")
    elif dataset_path.split("/")[-1] == "test":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", default="./filelists/aishell3_audio_sid_text_")
    parser.add_argument("--filelists", nargs="+",
                        default=["./data/AISHELL-3/train", "./data/AISHELL-3/test"])

    args = parser.parse_args()

    for filepath in args.filelists:
        new_filelist = args.outdir
        new_filelist += filepath.split("/")[-1] + "_filelist.txt"
        main(filepath, new_filelist)
