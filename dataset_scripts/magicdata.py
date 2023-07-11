import os
import argparse
import random
import json

from tqdm import tqdm

random.seed(1234)

speakers = {"G0051": "0", "G0052": "1", "G0053": "2",
            "G0054": "3", "G0055": "4", "G0067": "5",
            "G0068": "6", "G0070": "7", "G0071": "8",
            "G0072": "9", "G0001": "10", "G0002": "11",
            "G0003": "12", "G0004": "13", "G0005": "14",
            "G0006": "15", "G0007": "16", "G0008": "17",
            "G0009": "18", "G0010": "19",
            }

with open('data/magicdata/speakers_map.json', 'w') as outfile:
    json.dump(speakers, outfile)

def get_audiopath_textnorm(filepaths):
    all_lines = []
    for filepath in filepaths:
        print(f"Processing on {filepath}.")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                filetext = f.read()
            lines = filetext.split("\n")
            lines = lines[1:]
            random.shuffle(lines)
            all_lines.extend(lines)
        else:
            print(f"Could not find {filepath}!")

    for l in all_lines:
        if len(l) > 0:
            try:
                _, uid, sid, prompt, trans = l.split("\t")
                text = "[GD]" + trans + "[GD]"
                audio_path = "WAV/" + sid + "/" + uid
                audiopath_and_text = "|".join(
                    [audio_path, speakers[sid], text]) + "\n"
                yield audiopath_and_text
            except:
                print(f'Skipping line: {l}, length: {len(l)}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", default="./filelists/magicdata_audio_sid_text_train_filelist.txt")
    parser.add_argument("--filelists", nargs="+",
                        default=["./data/magicdata/UTTRANSINFO.txt", "./data/magicdata/UTTERANCEINFO.txt"])
    args = parser.parse_args()

    with open(args.outdir, "w", encoding="utf-8") as f:
        f.writelines(get_audiopath_textnorm(args.filelists))
