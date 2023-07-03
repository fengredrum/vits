import os
import re
import argparse
import random

from tqdm import tqdm

speakers = {"G0051": "0", "G0052": "1", "G0053": "2",
            "G0054": "3", "G0055": "4", "G0067": "5",
            "G0068": "6", "G0070": "7", "G0071": "8",
            "G0072": "9"}


def get_audiopath_textnorm(filepath):
    with open(filepath, "r") as f:
        filetext = f.read()
    lines = filetext.split("\n")
    lines = lines[1:]
    random.seed(1234)
    random.shuffle(lines)

    for l in lines:
        if len(l) > 0:
            try:
                _, uttrans_id, speaker_id, prompt, transcription = l.split(
                    "\t")
                text = "[GD]" + transcription + "[GD]"
                audio_path = "WAV/" + speaker_id + "/" + uttrans_id
                audiopath_and_text = "|".join(
                    [audio_path, speakers[speaker_id], text]) + "\n"
                yield audiopath_and_text
            except:
                print(f'Skipping line: {l}, length: {len(l)}.')


def main(dataset_path, output_dir):
    # Obtain the dataset directories and collect tsv files
    print(f"Processing on {dataset_path}.")

    if os.path.exists(dataset_path):
        with open(output_dir, "w", encoding="utf-8") as f:
            f.writelines(get_audiopath_textnorm(dataset_path))
    else:
        print(f"Could not find {dataset_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", default="./filelists/magicdata_audio_sid_text_")
    parser.add_argument("--filelists", nargs="+",
                        default=["./data/magicdata/UTTRANSINFO.txt", ])

    args = parser.parse_args()

    for filepath in args.filelists:
        new_filelist = args.outdir
        new_filelist += "train_filelist.txt"
        main(filepath, new_filelist)
