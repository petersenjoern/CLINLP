#%%
import pathlib
import torch
from utils import LabelSet, TraingDataset, TraingingBatch
from transformers import AutoTokenizer, AdamW, BertForTokenClassification
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile as tprofiler
from torch.profiler import schedule, tensorboard_trace_handler
from preprocess import convert_tsv_to_conll_format
import torch.nn.functional as F

# configuration
BATCH_SIZE = 256
EPOCHS = 20
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
#%%
# Load data
PATH_BASE = pathlib.Path.cwd()
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

model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(trainset.label_set.ids_to_label.values())).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=5e-6)

trainloader = DataLoader(
    trainset,
    collate_fn=TraingingBatch,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

testloader = DataLoader(
    testset,
    collate_fn=TraingingBatch,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


def train(batch):
    """ training steps for each batch"""
    inputs, masks, labels = batch.input_ids, batch.attention_masks, batch.labels
    outputs = model(
        input_ids=inputs,
        attention_mask=masks,
        labels=labels,
    )
    outputs.loss.backward()
    optimizer.step()
    logits = outputs.logits
    # logits = outputs.logits.detach().cpu().numpy()
    # label_ids = labels.to('cpu').numpy()
    return logits, labels, outputs.loss




with tprofiler(
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler('./data/tensorboard/log/BertForTokenClassification'),
        record_shapes=True,
        with_stack=True
) as prof:
    predictions , true_labels = [], []
    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        for step, batch_data in enumerate(trainloader):
            logits, labels, loss_value = train(batch_data)
            predictions.append(logits)
            true_labels.append(labels)
            prof.step()
            # print(F.softmax(torch.tensor(logits), dim=-1))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))

            if (loss_value <= 1.0) or epoch >= 7:
                break
