import pathlib
from pandas import read_csv

PATH_BASE = pathlib.Path.cwd()
PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")

PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.tsv")

def convert_tsv_to_csv(path_tsv, path_csv):
    # write comma-delimited file (comma is the default delimiter)
    df = read_csv(path_tsv, sep="\t")
    df.to_csv(path_csv, index=False)

def main():
    for file_in_dir in PATH_DATA_NER.iterdir():
        if str(file_in_dir).endswith(".tsv"):
            path_to_csv = str(file_in_dir).replace(".tsv", ".csv")
            convert_tsv_to_csv(file_in_dir, path_to_csv)

if __name__ == "__main__":
    main()