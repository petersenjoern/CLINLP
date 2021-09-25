import itertools
import torch

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Union, Any
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from collections import defaultdict

class LabelSet:
    """ Align target labels with tokens"""
    def __init__(self, labels: List[str]):
        """ Create BILU target labels"""
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l
        # Add the OUTSIDE label - no label for the token
    
    def __getitem__(self, item):
        return getattr(self, item)

    def align_tokens_and_annotations_bilou(self, tokenized, annotations):
        tokens = tokenized.tokens
        aligned_labels = ["O"] * len(
            tokens
        )  # Make a list to store our labels the same length as our tokens
        for anno in annotations:
            annotation_token_ix_set = (
                set()
            )  # A set that stores the token indices of the annotation
            for char_ix in range(anno["start"], anno["end"]):

                token_ix = tokenized.char_to_token(char_ix)
                if token_ix is not None:
                    annotation_token_ix_set.add(token_ix)
            if len(annotation_token_ix_set) == 1:
                # If there is only one token
                token_ix = annotation_token_ix_set.pop()
                prefix = (
                    "U"  # This annotation spans one token so is prefixed with U for unique
                )
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"

            else:

                last_token_in_anno_ix = len(annotation_token_ix_set) - 1
                for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                    if num == 0:
                        prefix = "B"
                    elif num == last_token_in_anno_ix:
                        prefix = "L"  # Its the last token
                    else:
                        prefix = "I"  # We're inside of a multi token annotation
                    aligned_labels[token_ix] = f"{prefix}-{anno['tag']}"
        return aligned_labels

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        raw_labels = self.align_tokens_and_annotations_bilou(tokenized_text, annotations)    
        return list(map(self.labels_to_id.get, raw_labels))



@dataclass
class TrainingExample:
    input_ids: List[int]
    attention_masks: List[int]
    labels: List[int]

class TraingDataset(Dataset):
    def __init__(
        self,
        data: Any,
        label_set: LabelSet, #BILU target labels
        tokenizer: AutoTokenizer,
        tokens_per_batch=32,
        window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        for example in data:
            # changes tag key to label
            for a in example["annotations"]:
                a["label"] = a["tag"]
        self.texts = []
        self.annotations = []

        # Move up in loop above
        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )
            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        empty_label_id = "O"
        for encoding, label in zip(tokenized_batch.encodings, aligned_labels):
            length = len(label)  # How long is this sequence
            for start in range(0, length, self.window_stride):

                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                        + [self.tokenizer.pad_token_id]
                        * padding_to_add,  # padding if needed
                        labels=(
                            label[start:end]
                            + [-100] * padding_to_add  # padding if needed
                        ),  # -100 is a special token for padding of labels,
                        attention_masks=(
                            encoding.attention_mask[start:end]
                            + [0]
                            * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:
        return self.training_examples[idx]

class TraingingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        input_ids: List[int] = []
        masks: List[int] = []
        labels: List[int] = []
        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)


def prepare_batch_for_metrics(batch: TraingingBatch, predictions:torch.Tensor) -> Tuple[List[int], List[int]]:
    # get the sentence lengths
    s_lengths = batch.attention_masks.sum(dim=1)
    # iterate through the examples
    batch_true_values = []
    batch_pred_values = []
    for idx, length in enumerate(s_lengths):
        # get the true values
        true_values = batch.labels[idx][:length].tolist()
        batch_true_values.extend(true_values)
        # get the predicted values
        pred_values = torch.argmax(predictions, dim=2)[idx][:length].tolist()
        batch_pred_values.extend(pred_values)
    return batch_true_values, batch_pred_values

def bilu_to_non_bilu(iterat: Iterator) -> Dict[str, List[int]]:
    """Prepare non BILU labels mapping to ids"""
    tally = defaultdict(list)
    for i,item in enumerate(iterat):
        tally[item].append(i)
    return dict([(key,locs) for key,locs in tally.items()])

@dataclass
class BiluMappings:
    non_bilu_label_to_bilu_ids: Dict[str, Tuple[List[int], int]]
    non_bilu_label_to_id:  Dict[str, int]

# Tuple[Dict[int, Tuple[List[int], int]], Dict[str, int]]
def ids_to_non_bilu_label_mapping(labelset: LabelSet) -> BiluMappings:
    """Mapping from ids to BILU and non-BILU mapping. This is used to remove the BILU labels to regular labels"""
    target_names = list(labelset["ids_to_label"].values())
    wo_bilu = [bilu_label.split("-")[-1] for bilu_label in target_names]
    non_bilu_mapping = bilu_to_non_bilu(wo_bilu)

    BiluMappings.non_bilu_label_to_bilu_ids = {}
    BiluMappings.non_bilu_label_to_id = {}
    for target_name, labels_list in non_bilu_mapping.items():
        # 'upper_bound': ([1, 2, 3, 4], 1)
        BiluMappings.non_bilu_label_to_bilu_ids[target_name] = labels_list, labels_list[0]
        # 'upper_bound': 1
        BiluMappings.non_bilu_label_to_id[target_name] = labels_list[0]
    
    return BiluMappings

def remove_bilu_ids_from_true_and_pred_values(non_bilu_label_to_bilu_ids: Dict[int, Tuple[List[int], int]], 
    true_values: List[int], pred_values: List[int]) -> Tuple[List[int], List[int]]:
    """Reduce the BILI ids (true and predicted) to regular labels and ids"""

    for idx, label in enumerate(true_values):
        for _, (labels_list, non_bilu_label) in non_bilu_label_to_bilu_ids.items():
            if label in labels_list:
                true_values[idx] = non_bilu_label
    
    for idx, label in enumerate(pred_values):
        for _, (labels_list, non_bilu_label) in non_bilu_label_to_bilu_ids.items():
            if label in labels_list:
                pred_values[idx] = non_bilu_label

    return true_values, pred_values


def get_multilabel_metrics(true_values: List[int], pred_values: List[int],
    non_bilu_label_to_bilu_ids: Dict[str, Tuple[List[int], int]], non_bilu_label_to_id: Dict[str, int],
    labelset: LabelSet, remove_bilu: bool = True) -> Dict[str, Dict[str, Union[float, int]]]:
    """Create a classification report for all labels in the dataset"""
    
    if remove_bilu:
        labels = list(non_bilu_label_to_id.keys())
        target_names = list(non_bilu_label_to_id.values())
        true_values, pred_values = remove_bilu_ids_from_true_and_pred_values(non_bilu_label_to_bilu_ids, true_values, pred_values)
    else:
        labels = list(labelset["ids_to_label"].keys())
        target_names = list(labelset["ids_to_label"].values())

    metrics = classification_report(
        y_true=true_values, y_pred=pred_values, 
        labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )
    return metrics

def get_confusion_matrix(num_labels:int, normalize:bool, batch) -> torch.Tensor:
        """Evaluate a batch with a confusion matrix"""
        confusion_matrix = torch.zeros(num_labels, num_labels)
        # get the sentence lengths
        s_lengths = batch.attention_masks.sum(dim=1)
        # iterate through the examples
        for idx, length in enumerate(s_lengths):
            # get the true values
            true_values = batch.labels[idx][:length]
            # get the predicted values
            pred_values = torch.argmax(outputs[1], dim=2)[idx][:length]
            # go through all true and predicted values and store them in the confusion matrix
            for true, pred in zip(true_values, pred_values):
                confusion_matrix[true.item()][pred.item()] += 1
            if normalize:
                # Normalize by dividing every row by its sum
                for i in range(num_labels):
                    confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()
        print(torch.diagonal(confusion_matrix, 0))
        return confusion_matrix
