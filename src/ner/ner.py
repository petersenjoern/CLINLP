#%%
import pathlib
from typing import Dict, Iterator, List
import torch
from utils import LabelSet, TraingDataset, TraingingBatch
from transformers import AutoTokenizer, AdamW, BertForTokenClassification
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile as tprofiler
from torch.profiler import schedule, tensorboard_trace_handler
from preprocess import convert_tsv_to_conll_format
from sklearn.metrics import classification_report
from collections import defaultdict

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

def prepare_batch_for_metrics(batch: TraingingBatch, predictions:torch.Tensor):
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

def remove_bilu(bilu_labels: List[str], true_values: List[int], pred_values: List[int]):
    """Remove the BILU tagging of the labels"""
    wo_bilu = [bilu_label.split("-")[-1] for bilu_label in bilu_labels]
    non_bilu_mapping = bilu_to_non_bilu(wo_bilu)

    non_bilu_mapping = {}
    non_bilu_target_to_label = {}
    for target_name, labels_list in non_bilu_mapping.items():
        # 'upper_bound': ([1, 2, 3, 4], 1)
        non_bilu_mapping[target_name] = labels_list, labels_list[0]
        # 'upper_bound': 1
        non_bilu_target_to_label[target_name] = labels_list[0]
    
    for val in true_values:
        for _, (labels_list, non_bilu_value) in non_bilu_mapping.items():
            if val in labels_list:
                val = non_bilu_value
    
    for val in pred_values:
        for _, (labels_list, non_bilu_value) in non_bilu_mapping.items():
            if val in labels_list:
                val = non_bilu_value

    return list(non_bilu_target_to_label.keys()), list(non_bilu_target_to_label.values()), true_values, pred_values


def get_multilabel_metrics(true_values: List[int], pred_values: List[int], labelset: LabelSet):

        labels = list(labelset["ids_to_label"].keys())
        target_names = list(labelset["ids_to_label"].values())
        target_names, labels, true_values, pred_values = remove_bilu(target_names, true_values, pred_values)
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
        return confusion_matrix


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
        epoch_true_sample_values = []
        epoch_pred_sample_values = []
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
            
            # Log every 50 batches.
            if step % 50 == 0:
            #     confusion = get_confusion_matrix(num_labels=num_labels, normalize=True, batch=batch)
            #     print(torch.diagonal(confusion, 0))
                batch_true_values, batch_pred_values = prepare_batch_for_metrics(batch=batch, predictions=outputs[1])
                epoch_true_sample_values.extend(batch_true_values)
                epoch_pred_sample_values.extend(batch_pred_values)
                metrics = get_multilabel_metrics(epoch_true_sample_values, epoch_pred_sample_values, label_set_train)


        # update the model one last time for this epoch
        optimizer.step()
        optimizer.zero_grad()
        metrics = get_multilabel_metrics(epoch_true_sample_values, epoch_pred_sample_values, label_set_train)
        print(metrics)


# Evaluate on test dataset
model = model.eval()
epoch_true_sample_values = []
epoch_pred_sample_values = []
for step, batch in enumerate(testloader):
    # do not calculate the gradients
    with torch.no_grad():
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
        batch_true_values, batch_pred_values = prepare_batch_for_metrics(batch=batch, predictions=outputs[1])
        epoch_true_sample_values.extend(batch_true_values)
        epoch_pred_sample_values.extend(batch_pred_values)
metrics = get_multilabel_metrics(epoch_true_sample_values, epoch_pred_sample_values, label_set_train)
print(metrics)