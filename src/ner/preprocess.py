import pathlib
from pandas import read_csv

PATH_BASE = pathlib.Path.cwd()
PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")

PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.tsv")

def convert_tsv_to_csv(path_tsv):
    # write comma-delimited file (comma is the default delimiter)
    df = read_csv(path_tsv, sep="\t")
    path_csv = str(path_tsv).replace(".tsv", ".csv")
    df.to_csv(path_csv, index=False)

def convert_to_conll_format():
    pass

def main():
    for file_in_dir in [PATH_DATA_NER_TRAIN]:
        if str(file_in_dir).endswith(".tsv"):
            df = read_csv(file_in_dir, sep="\t")
    if not df.any():
        raise Exception("Couldnt load data")
        

if __name__ == "__main__":
    main()