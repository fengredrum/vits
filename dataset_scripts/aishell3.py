import os
import argparse
import json
import random

from tqdm import tqdm

random.seed(1234)
speakers = {}


def get_audiopath_textnorm(filepath, filename):
    with open(filepath + "/" + filename, "r") as f:
        filetext = f.read()
    lines = filetext.split("\n")
    lines = lines[5:]
    random.shuffle(lines)

    sid = -1
    for l in lines:
        if len(l) > 1:
            try:
                audio_path, pinyin, text = l.split("|")
                text = "[ZH]" + text + "[ZH]"
                speaker_id = audio_path[:7]
                if speaker_id not in speakers:
                    sid += 1
                    speakers.update({speaker_id: str(sid)})

                audio_path = filepath + "/wav/" + speaker_id + "/" + audio_path + ".wav"
                audiopath_and_text = "|".join(
                    [audio_path, speakers[speaker_id], text]) + "\n"
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

    with open('data/AISHELL-3/speakers_map.json', 'w') as outfile:
        json.dump(speakers, outfile)
