import pathlib
from typing import List, Any
from transformers import AutoTokenizer, AutoModel, BatchEncoding,AdamW
from tokenizers import Encoding
from datasets import load_dataset
import itertools
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import DataLoader



sequence_a = " @NUMBER years of age or older with resectable pancreatic adenocarcinoma for whom surgery is planned ( includes borderline resectable if deemed appropriate by surgical investigators ) or has occurred within the past @NUMBER years"
sequence_a_annotations = [
    dict(start=1,end=14,text="@NUMBER",tag="lower_bound"),
    dict(start=18,end=21,text="age",tag="age"),
    dict(start=36,end=72,text="resectable pancreatic adenocarcinoma",tag="cancer"),
    dict(start=82,end=89,text="surgery",tag="treatment"),
    dict(start=211,end=229,text="past @NUMBER years",tag="upper_bound"),
]
sequence_b = " severe mood disorder ( phq8 > @NUMBER )"
sequence_b_annotations = [
    dict(start=8,end=21,text="mood disorder",tag="chronic_disease"),
    dict(start=24,end=28,text="phq8",tag="clinical_variable"),
    dict(start=31,end=38,text="@NUMBER",tag="lower_bound"),
]

raw = [{"content": sequence_a, "annotations": sequence_a_annotations}, {"content": sequence_b, "annotations": sequence_b_annotations}, {"content": "this is some text", "annotations": []}]


PATH_BASE = pathlib.Path.cwd()
PATH_DATA_CTGOV_TRIALS = PATH_BASE.joinpath("data", "ctgov", "extraction", "clinical_trials_similar.csv")
PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")
PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.csv")

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
#dataset = load_dataset("wnut_17")
#print(dataset["train"][0])
#train_encodings = tokenizer(dataset["train"]["tokens"], is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
train_encodings: BatchEncoding = tokenizer([sequence_a, sequence_b], is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True)
print(dir(train_encodings[1]))
print(train_encodings[1].ids)
print(train_encodings[1].tokens)

## tokens with tags alignment
tokenized_text: Encoding  = train_encodings[1]

def align_tokens_and_annotations_bilou(tokenized, annotations):
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

labels = align_tokens_and_annotations_bilou(tokenized_text, sequence_b_annotations)
for token, label in zip(tokenized_text.tokens, labels):
    print(token, "-", label)



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

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        raw_labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)    
        return list(map(self.labels_to_id.get, raw_labels))


example_label_set = LabelSet(labels=["lower_bound", "age", "cancer", "treatment", "upper_bound", "chronic_disease", "clinical_variable"])
aligned_label_ids = example_label_set.get_aligned_label_ids_from_annotations(
    tokenized_text, sequence_b_annotations
)

for token, label in zip(tokenized_text.tokens, aligned_label_ids):
    print(token, "-", label)

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

label_set = LabelSet(labels=["lower_bound", "age", "cancer", "treatment", "upper_bound", "chronic_disease", "clinical_variable"])
ds = TraingDataset(
    data=raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=16
)
ex = ds[1]
print("xxx" * 20)
print(ex)



class TraingingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []
        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)

model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(ds.label_set.ids_to_label.values()))
optimizer = AdamW(model.parameters(), lr=5e-6)

dataloader = DataLoader(
    ds,
    collate_fn=TraingingBatch,
    batch_size=4,
    shuffle=True,
)
for num, batch in enumerate(dataloader):
    loss, logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_masks,
        labels=batch.labels,
    )
    loss.backward()
    optimizer.step()
    print(loss)

# data = load_dataset('csv', data_files={"train": [str(PATH_DATA_NER_TRAIN)]})
# print(dir(data))
# print(len(data["train"]))
# print(data["train"][0])
exit(-1)


# Tokenize and covert into ids
tokenized_sequence = tokenizer.tokenize(sequence_a)
print(tokenized_sequence)
inputs = tokenizer(sequence_a)
encoded_sequence = inputs["input_ids"]
print(encoded_sequence)

# tokenize, covert to ids, padding and attention masking (which ids to pay attention to)
padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
print(padded_sequences)

inputs = tokenizer([sequence_a, sequence_b], padding="max_length", truncation=True)
print(inputs)
