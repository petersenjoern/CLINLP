import pathlib
from utils import LabelSet, TraingDataset, TraingingBatch
from transformers import AutoTokenizer, AdamW, BertForTokenClassification
from torch.utils.data.dataloader import DataLoader
from preprocess import convert_tsv_to_conll_format


PATH_BASE = pathlib.Path.cwd()
PATH_DATA_CTGOV_TRIALS = PATH_BASE.joinpath("data", "ctgov", "extraction", "clinical_trials_similar.csv")
PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")
PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.tsv")

data = convert_tsv_to_conll_format([PATH_DATA_NER_TRAIN])
data_train = data[0]
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
label_set_train = LabelSet(labels=data_train.unique_labels)
ds = TraingDataset(
    data=data_train.preprocessed, tokenizer=tokenizer, label_set=label_set_train, tokens_per_batch=32
)

model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(ds.label_set.ids_to_label.values()))
optimizer = AdamW(model.parameters(), lr=5e-6)

dataloader = DataLoader(
    ds,
    collate_fn=TraingingBatch,
    batch_size=32,
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
