import itertools
import torch

from dataclasses import dataclass
from typing import List, Any
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class LabelSet:
    def __init__(self, labels: List[str]):
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
        label_set: LabelSet,
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