import pathlib
from utils import LabelSet, TraingDataset, TraingingBatch
from transformers import AutoTokenizer, AdamW, BertForTokenClassification
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile as tprofiler
from torch.profiler import schedule, tensorboard_trace_handler
from preprocess import convert_tsv_to_conll_format


# Load data
PATH_BASE = pathlib.Path.cwd()
PATH_DATA_CTGOV_TRIALS = PATH_BASE.joinpath("data", "ctgov", "extraction", "clinical_trials_similar.csv")
PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")
PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.tsv")
PATH_DATA_NER_TEST = PATH_DATA_NER.joinpath("test_processed_medical_ner.tsv")


data = convert_tsv_to_conll_format([PATH_DATA_NER_TRAIN, PATH_DATA_NER_TEST])
data_train = data[0]
data_test = data[1]

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
label_set_train = LabelSet(labels=data_train.unique_labels) # unique labels are expected to be same
trainset = TraingDataset(
    data=data_train.preprocessed, tokenizer=tokenizer, label_set=label_set_train, tokens_per_batch=32
)
testset = TraingDataset(
    data=data_test.preprocessed, tokenizer=tokenizer, label_set=label_set_train, tokens_per_batch=32
)

model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(trainset.label_set.ids_to_label.values()))
optimizer = AdamW(model.parameters(), lr=5e-6)

trainloader = DataLoader(
    trainset,
    collate_fn=TraingingBatch,
    batch_size=4,
    shuffle=True,
)

testloader = DataLoader(
    testset,
    collate_fn=TraingingBatch,
    batch_size=4,
    shuffle=True,
)


def train(data):
    """ training steps for each batch"""
    inputs, masks, labels = data.input_ids, data.attention_masks, data.labels
    outputs = model(
        input_ids=inputs,
        attention_mask=masks,
        labels=labels,
    )
    outputs.loss.backward()
    optimizer.step()
    print(outputs.loss)



with tprofiler(
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler('./data/tensorboard/log/BertForTokenClassification'),
        record_shapes=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(trainloader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data)
        prof.step()



# save
# for num, batch in enumerate(trainloader):
#     outputs = model(
#         input_ids=batch.input_ids,
#         attention_mask=batch.attention_masks,
#         labels=batch.labels,
#     )
#     outputs.loss.backward()
#     optimizer.step()
#     print(outputs.loss)
