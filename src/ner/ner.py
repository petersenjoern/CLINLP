import pathlib
from utils import LabelSet, TraingDataset, TraingingBatch
from transformers import AutoTokenizer, BatchEncoding, AdamW, BertForTokenClassification
from tokenizers import Encoding
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


# PATH_BASE = pathlib.Path.cwd()
# PATH_DATA_CTGOV_TRIALS = PATH_BASE.joinpath("data", "ctgov", "extraction", "clinical_trials_similar.csv")
# PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")
# PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.csv")

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
train_encodings: BatchEncoding = tokenizer([sequence_a, sequence_b], is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True)
print(dir(train_encodings[1]))
print(train_encodings[1].ids)
print(train_encodings[1].tokens)

## tokens with tags alignment
tokenized_text: Encoding  = train_encodings[1]
example_label_set = LabelSet(labels=["lower_bound", "age", "cancer", "treatment", "upper_bound", "chronic_disease", "clinical_variable"])
aligned_label_ids = example_label_set.get_aligned_label_ids_from_annotations(
    tokenized_text, sequence_b_annotations
)
for token, label in zip(tokenized_text.tokens, aligned_label_ids):
    print(token, "-", label)





label_set = LabelSet(labels=["lower_bound", "age", "cancer", "treatment", "upper_bound", "chronic_disease", "clinical_variable"])
ds = TraingDataset(
    data=raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=16
)
ex = ds[1]
print("xxx" * 20)
print(ex)





model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(ds.label_set.ids_to_label.values()))
optimizer = AdamW(model.parameters(), lr=5e-6)

dataloader = DataLoader(
    ds,
    collate_fn=TraingingBatch,
    batch_size=4,
    shuffle=True,
)
for num, batch in enumerate(dataloader):
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_masks,
        labels=batch.labels,
    )
    outputs.loss.backward()
    optimizer.step()
    print(outputs.loss)
