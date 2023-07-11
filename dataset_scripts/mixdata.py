import os
import re
import argparse
import random
import json

from tqdm import tqdm

random.seed(1234)


def get_audiopath_textnorm(filelist, speaker_map):
    total_lines = []
    for filepath in filelist:
        with open(filepath, "r") as f:
            filetext = f.read()
        lines = filetext.split("\n")
        total_lines.extend(lines)

    random.shuffle(total_lines)
    for l in total_lines:
        if len(l) > 0:
            try:
                audio_path, _, text = l.split("|")
                speaker = audio_path.split("/")[-2]
                if speaker in speaker_map:
                    audiopath_and_text = "|".join(
                        [audio_path, speaker_map[speaker], text]) + "\n"
                    yield audiopath_and_text
            except:
                print(f'Skipping line: {l}, length: {len(l)}.')


def main(args):
    speaker_list = []
    for i in range(len(args.speaker_maps)):
        with open(args.speaker_maps[i], "r") as f:
            filedict = f.read()
        speaker_map = json.loads(filedict)
        speakers = list(speaker_map.keys())
        random.shuffle(speakers)
        speaker_list.extend(speakers[:args.num_speakers[i]])

    speaker_dict = {}
    sid = 0
    for speaker in speaker_list:
        speaker_dict.update({speaker: str(sid)})
        sid += 1

    with open(args.dump_path, 'w') as outfile:
        json.dump(speaker_dict, outfile)

    with open(args.outdir, "w", encoding="utf-8") as f:
        f.writelines(get_audiopath_textnorm(args.filelists, speaker_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", default="./filelists/mix_audio_sid_text_train_filelist.txt")
    parser.add_argument(
        "--dump_path", default="./data/mixdata/speakers_map.json")
    parser.add_argument("--speaker_maps", nargs="+",
                        default=["./data/AISHELL-3/speakers_map.json", "./data/magicdata/speakers_map.json"])
    parser.add_argument("--num_speakers", nargs="+",
                        default=[80, 20])
    parser.add_argument("--filelists", nargs="+",
                        default=["filelists/aishell3_audio_sid_text_train_filelist.txt", "filelists/magicdata_audio_sid_text_train_filelist.txt"])

    args = parser.parse_args()
    main(args)
