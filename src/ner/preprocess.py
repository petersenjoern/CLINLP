import pathlib
from pandas import read_csv
from typing import List, Dict, Any, Tuple

from pandas.core.frame import DataFrame
from dataclasses import dataclass

def convert_tsv_to_csv(path_tsv):
    """Convert a tsv to a csv and save it"""
    df = read_csv(path_tsv, sep="\t")
    path_csv = str(path_tsv).replace(".tsv", ".csv")
    df.to_csv(path_csv, index=False)


@dataclass
class AnnotationsAndLabelsStruct:
    preprocessed: List[Dict[str, Any]]
    unique_labels: List[str]
 
def convert_tsv_to_conll_format(datasets: List[pathlib.Path]) -> AnnotationsAndLabelsStruct:
    """ Convert TSV annotations and content to List[Dict]"""
    preprocessed_all = []
    for file_in_dir in datasets:
        if str(file_in_dir).endswith(".tsv"):
            df = read_csv(file_in_dir, sep="\t")

            # set columns and prepare annotations
            df.columns = ["labels", "content"]
            preprocessed, unique_labels = prepare_examples_to_content_and_annotations(df)
            preprocessed_all.append(AnnotationsAndLabelsStruct(preprocessed, unique_labels))
    return preprocessed_all
        
def prepare_examples_to_content_and_annotations(df: DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    """ Preprocessed each row in the DF to its content and annotations"""
    preprocssed_data = []
    unique_labels = set()
    for _, row in df.iterrows():
        labels_list = str(row["labels"]).split(",")
        content = str(" " + row["content"]) #start end idx are made for text starting with space
        
        annotations = []
        for labels in labels_list:
            start, end, tag = tuple(labels.strip().split(":"))
            start, end, tag = int(start), int(end), str(tag)
            text = content[start:end]
            unique_labels.add(tag)
            annotations.append(dict(start=start, end=end, text=text, tag=tag))
        preprocessed_example = {"content": content, "annotations": annotations}
        preprocssed_data.append(preprocessed_example)
    return preprocssed_data, unique_labels

if __name__ == "__main__":
    PATH_BASE = pathlib.Path.cwd()
    PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")
    PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.tsv")
    convert_tsv_to_conll_format([PATH_DATA_NER_TRAIN])