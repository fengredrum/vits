import os
import re
from tqdm import tqdm


def get_audiopath_textnorm(files_list, root):
    for i in tqdm(range(len(files_list))):
        with open(files_list[i], "r") as f:
            filetext = f.read()
        lines = filetext.split("\n")

        for l in lines:
            if len(l) > 1:
                try:
                    audio_path, text, text_norm = l.split("\t")
                    audio_path = re.sub("_", "/", audio_path,
                                        1).split("_")[0] + "/" + audio_path + ".wav"
                    audiopath_and_text_norm = "|".join(
                        [audio_path, text_norm]) + "\n"
                    yield root + "/" + audiopath_and_text_norm
                except:
                    print(f'Skipping line: {l}, length: {len(l)}.')


def main(dataset_path, output_dir):
    # Obtain the dataset directories and collect tsv files
    print(f"Processing on {dataset_path}.")

    trans_paths = []
    for path, subdirs, files in os.walk(dataset_path):
        for name in files:
            splited_name = name.split(".")
            if splited_name[-1] == "tsv" and splited_name[-2] == "trans":
                trans_paths.append(os.path.join(path, name))

    if len(trans_paths) > 0:
        with open(output_dir, "w", encoding="utf-8") as f:
            f.writelines(get_audiopath_textnorm(trans_paths, dataset_path))
    else:
        print(f"Could not find {dataset_path}.")


if __name__ == "__main__":
    paths = ["./data/LibriTTS/dev-clean",]
    for path in paths:
        new_filelist = "./filelists/libritts_audio_text_"
        new_filelist += path.split("/")[-1] + ".txt"
        main(path, new_filelist)
