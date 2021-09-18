#%%
import pathlib
from typing import List
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
model.config.id2label = label_set_train["ids_to_label"]
model.config.label2id = label_set_train["labels_to_id"]
num_labels = len(label_set_train["ids_to_label"])
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


with tprofiler(
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler('./data/tensorboard/log/BertForTokenClassification'),
        record_shapes=True,
        with_stack=True
) as prof:
    train_loss = []
    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        current_loss = 0
        for step, batch in enumerate(trainloader):
            # move the batch tensors to the same device as the model
            batch.attention_masks = batch.attention_masks.to(DEVICE)
            batch.input_ids = batch.input_ids.to(DEVICE)
            batch.labels = batch.labels.to(DEVICE)
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_masks,
                labels=batch.labels,
            )
            # the outputs are of shape (loss, logits)
            loss = outputs[0]
            # with the .backward method it calculates all 
            # of  the gradients used for autograd
            loss.backward()
            # NOTE: if we append `loss` (a tensor) we will force the GPU to save
            # the loss into its memory, potentially filling it up. To avoid this
            # we rather store its float value, which can be accessed through the
            # `.item` method
            current_loss += loss.item()
            if step % 8 == 0 and step > 0:
                # update the model using the optimizer
                optimizer.step()
                # once we update the model we set the gradients to zero
                optimizer.zero_grad()
                # store the loss value for visualization
                train_loss.append(current_loss / 32)
                print(current_loss)
                current_loss = 0
            
            # Log every 200 batches.
            if step % 200 == 0:
                confusion = torch.zeros(num_labels, num_labels)
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
                        confusion[true.item()][pred.item()] += 1
                    # Normalize by dividing every row by its sum
                    for i in range(num_labels):
                        confusion[i] = confusion[i] / confusion[i].sum()


        # update the model one last time for this epoch
        optimizer.step()
        optimizer.zero_grad()
        # visual check after each epoch to see if we are improving in predicting the different cats
        print(torch.diagonal(confusion, 0))