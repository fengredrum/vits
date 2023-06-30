import os
import re
import argparse

from tqdm import tqdm


def get_audiopath_textnorm(files_list, root):
    for i in tqdm(range(len(files_list))):
        sid = str(files_list[i][1])
        with open(files_list[i][0], "r") as f:
            filetext = f.read()
        lines = filetext.split("\n")

        for l in lines:
            if len(l) > 1:
                try:
                    audio_path, text, text_norm = l.split("\t")
                    audio_path = re.sub("_", "/", audio_path,
                                        1).split("_")[0] + "/" + audio_path + ".wav"
                    audiopath_and_text_norm = "|".join(
                        [audio_path, sid, text_norm]) + "\n"
                    yield root + "/" + audiopath_and_text_norm
                except:
                    print(f'Skipping line: {l}, length: {len(l)}.')


def main(dataset_path, output_dir):
    # Obtain the dataset directories and collect tsv files
    print(f"Processing on {dataset_path}.")

    trans_paths = []
    sid = -1
    reader_set = set()
    for path, subdirs, files in os.walk(dataset_path):
        for name in files:
            splited_name = name.split(".")
            if splited_name[-1] == "tsv" and splited_name[-2] == "trans":
                reader_id = path.split("/")[-2]
                if reader_id not in reader_set:
                    sid += 1
                    reader_set.add(reader_id)
                trans_paths.append([os.path.join(path, name), sid])

    if len(trans_paths) > 0:
        with open(output_dir, "w", encoding="utf-8") as f:
            f.writelines(get_audiopath_textnorm(trans_paths, dataset_path))
    else:
        print(f"Could not find {dataset_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", default="./filelists/libritts_audio_sid_text_")
    parser.add_argument("--filelists", nargs="+",
                        default=["./data/LibriTTS/dev-clean", "./data/LibriTTS/train-clean-100"])

    args = parser.parse_args()

    for filepath in args.filelists:
        args.outdir += filepath.split("/")[-1] + "_filelist.txt"
        main(filepath, args.outdir)
