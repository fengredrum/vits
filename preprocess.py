import multiprocessing
import argparse
import text
from tqdm import tqdm
from utils import load_filepaths_and_text


def clean_text(path_and_text, text_index, cleaners):
    raw_text = path_and_text[text_index]
    cleaned_text = text._clean_text(raw_text, cleaners)
    path_and_text[text_index] = cleaned_text
    return path_and_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="test")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--filelists", nargs="+",
                        default=["filelists/ljs_audio_text_train_filelist.txt", "filelists/ljs_audio_text_val_filelist.txt"])
    parser.add_argument("--text_cleaners", nargs="+",
                        default=["english_cleaners2"])
    parser.add_argument("--multiprocessing", default=True, type=bool)

    args = parser.parse_args()

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)

        if args.multiprocessing:
            pbar = tqdm(total=len(filepaths_and_text))
            pbar.set_description('Processing')
            pool = multiprocessing.Pool()
            results = []
            for data in filepaths_and_text:
                tmp = pool.apply_async(
                    clean_text, (data, args.text_index, args.text_cleaners), callback=lambda *stats: pbar.update())
                results.append(tmp)
            pool.close()
            pool.join()
            filepaths_and_text = [line.get() for line in results]
        else:
            for i in tqdm(range(len(filepaths_and_text))):
                filepaths_and_text[i] = clean_text(
                    filepaths_and_text[i], args.text_index, args.text_cleaners)

        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
