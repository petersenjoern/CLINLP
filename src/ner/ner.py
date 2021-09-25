#%%
from pathlib import Path
import torch
from utils import LabelSet, TraingDataset, TraingingBatch, get_multilabel_metrics, ids_to_non_bilu_label_mapping, prepare_batch_for_metrics
from transformers import AutoTokenizer, AdamW, BertForTokenClassification, optimization
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile as tprofiler
from torch.profiler import schedule, tensorboard_trace_handler
from preprocess import convert_tsv_to_conll_format


# configuration
BATCH_SIZE = 256
EPOCHS = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PATH_BASE = Path.cwd()
PATH_DATA_NER = PATH_BASE.joinpath("data", "ner")
PATH_DATA_NER_TRAIN = PATH_DATA_NER.joinpath("train_processed_medical_ner.tsv")
PATH_DATA_NER_TEST = PATH_DATA_NER.joinpath("test_processed_medical_ner.tsv")
PATH_FINEDTUNED_MODEL = PATH_DATA_NER.joinpath("model", "finedtuned_bert")
PATH_TENSORBOARD_LOGS = PATH_BASE.joinpath("data", "tensorbard", "log", "BertForTokenClassification")


def train(model: BertForTokenClassification, optimizer: optimization, dataloader: DataLoader,
    epochs: int, bilu:bool, tensorboard_logs:Path, save_directory:Path=None, ):
    with tprofiler(
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler(tensorboard_logs),
            record_shapes=True,
            with_stack=True
    ) as prof:
        train_loss = []
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            current_loss = 0
            epoch_true_sample_values = []
            epoch_pred_sample_values = []
            for step, batch in enumerate(dataloader):
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
                    batch_true_values, batch_pred_values = prepare_batch_for_metrics(batch=batch, predictions=outputs[1])
                    epoch_true_sample_values.extend(batch_true_values)
                    epoch_pred_sample_values.extend(batch_pred_values)

            # update the model one last time for this epoch
            optimizer.step()
            optimizer.zero_grad()
            # visually inspect if metrics are improving over time
            metrics = get_multilabel_metrics(
                epoch_true_sample_values,
                epoch_pred_sample_values,
                non_bilu_label_to_bilu_ids,
                non_bilu_label_to_id,
                label_set_train,
                bilu
            )
            print(metrics)
    if save_directory:
        model.save_pretrained(save_directory)
    return model
    

def evaluate(model: BertForTokenClassification, dataloader: DataLoader, bilu:bool):
    # Evaluate on test dataset
    model = model.eval() #equivalent to model.train(False)
    epoch_true_sample_values = []
    epoch_pred_sample_values = []
    for step, batch in enumerate(dataloader):
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
    metrics = get_multilabel_metrics(
        epoch_true_sample_values,
        epoch_pred_sample_values,
        non_bilu_label_to_bilu_ids,
        non_bilu_label_to_id,
        label_set_train,
        bilu
    )
    print(metrics)



# Execution
# Preparation
data = convert_tsv_to_conll_format([PATH_DATA_NER_TRAIN, PATH_DATA_NER_TEST])
data_train = data[0]
data_test = data[1]

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
label_set_train = LabelSet(labels=data_train.unique_labels) # unique labels are expected to be same
non_bilu_label_to_bilu_ids, non_bilu_label_to_id = ids_to_non_bilu_label_mapping(label_set_train)

trainset = TraingDataset(
    data=data_train.preprocessed, tokenizer=tokenizer, label_set=label_set_train, tokens_per_batch=32
)
testset = TraingDataset(
    data=data_test.preprocessed, tokenizer=tokenizer, label_set=label_set_train, tokens_per_batch=32
)

model_pretrained = BertForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(trainset.label_set.ids_to_label.values())).to(DEVICE)
model_pretrained.config.id2label = label_set_train["ids_to_label"]
model_pretrained.config.label2id = label_set_train["labels_to_id"]
num_labels = len(label_set_train["ids_to_label"])
optimizer = AdamW(model_pretrained.parameters(), lr=5e-6)

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

# Modeling
model_finetuned = train(model=model_pretrained, dataloader=trainloader, optimizer=optimizer,
    epochs=EPOCHS, bilu=False, tensorboard_logs=PATH_TENSORBOARD_LOGS, save_directory=PATH_FINEDTUNED_MODEL)

# Evaluation
evaluate(model=model_finetuned, dataloader=testloader, bilu=False)



#TODO: log metrics to tensorboard
#TODO: add hydra
#TODO: extract predictions and add to output file incl. input incl/exl. criteria
# %%