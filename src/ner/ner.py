import json
from pathlib import Path
from typing import Tuple
import torch
from omegaconf import DictConfig
import hydra
from utils import (LabelSet, TraingDataset, TraingingBatch, BiluMappings, get_multilabel_metrics,
    ids_to_non_bilu_label_mapping, prepare_batch_for_metrics)
from transformers import AutoTokenizer, AdamW, BertForTokenClassification, optimization
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile as tprofiler
from torch.profiler import schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from preprocess import AnnotationsAndLabelsStruct, convert_tsv_to_conll_format


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def train(cfg: DictConfig, model: BertForTokenClassification, optimizer: optimization, dataloader: DataLoader, labelset: LabelSet,
    bilu_mappings: BiluMappings, save_directory:Path=None) -> BertForTokenClassification:

    model_comment = (f"bsize: {cfg.hyperparams.batch_size},  lr: {cfg.hyperparams.learning_rate}, "
        f"epochs: {cfg.hyperparams.epochs}, shuffle: {cfg.hyperparams.shuffle} device: {DEVICE}")
    tb = SummaryWriter(log_dir=cfg.caching.tensorboard_metrics, filename_suffix="-summary", comment=model_comment)
    with tprofiler(
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler(cfg.caching.tensorboard_profiler),
            record_shapes=True,
            with_stack=True,
    ) as prof:
        for epoch in range(cfg.hyperparams.epochs):
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
                    print(current_loss)
                    current_loss = 0
                
                # Log every 50 batches.
                if step % 50 == 0:
                    batch_true_values, batch_pred_values = prepare_batch_for_metrics(batch=batch, predictions=outputs[1])
                    epoch_true_sample_values.extend(batch_true_values)
                    epoch_pred_sample_values.extend(batch_pred_values)
                
                # pytorch profiler needs to be notified at each steps end
                prof.step() 

            # update the model one last time for this epoch
            optimizer.step()
            optimizer.zero_grad()
            # visually inspect if metrics are improving over time
            metrics = get_multilabel_metrics(
                epoch_true_sample_values,
                epoch_pred_sample_values,
                bilu_mappings.non_bilu_label_to_bilu_ids,
                bilu_mappings.non_bilu_label_to_id,
                labelset,
                cfg.model.evaluation.bilu
            )
            # Log the metrics to every epoch
            tb.add_scalar("Loss", loss.item(), epoch)
            tb.add_scalar("Precision", metrics['weighted avg']["precision"], epoch)
            tb.add_scalar("Recall", metrics['weighted avg']["recall"], epoch)
            tb.add_scalar("F1-Score", metrics['weighted avg']["f1-score"], epoch)
    
    # record the final results with hyperparams used
    tb.add_hparams(
        {
            "lr": cfg.hyperparams.learning_rate, "bsize": cfg.hyperparams.batch_size,
            "epochs":cfg.hyperparams.epochs, "shuffle": cfg.hyperparams.shuffle},
        {
            "precision": metrics['weighted avg']["precision"],
            "recall": metrics['weighted avg']["recall"],
            "f1-score": metrics['weighted avg']["f1-score"],
            "loss": loss.item(),
        },
    )
    if save_directory:
        model.save_pretrained(save_directory)
    return model
    

def evaluate(cfg: DictConfig, model: BertForTokenClassification, dataloader: DataLoader, labelset:LabelSet,
    bilu_mappings: BiluMappings, load_directory:Path=None, save_directory:Path=None) -> None:
    
    if not model:
        if load_directory:
            try:
                model = torch.load(load_directory)
            except:
                raise Exception("Model couldnt be loaded under: %s", load_directory)
        
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
        bilu_mappings.non_bilu_label_to_bilu_ids,
        bilu_mappings.non_bilu_label_to_id,
        labelset,
        cfg.model.evaluation.bilu
    )
    if save_directory:
        with open(save_directory, 'w') as outfile:
            json.dump(metrics, outfile)
    print(metrics)


def load_and_prepare(cfg: DictConfig) -> Tuple[AnnotationsAndLabelsStruct, AnnotationsAndLabelsStruct, BiluMappings]:
    # Preparation
    path_main = Path(__file__).parents[2]
    data = convert_tsv_to_conll_format(
        [path_main.joinpath(Path(cfg.datasets.train)), path_main.joinpath(Path(cfg.datasets.test))])
    data_train = data[0]
    data_test = data[1]

    label_set_train = LabelSet(labels=data_train.unique_labels)
    bilu_mappings = ids_to_non_bilu_label_mapping(label_set_train)
    return data_train, data_test, label_set_train, bilu_mappings

@hydra.main(config_path=".", config_name="config_ner")
def main(cfg: DictConfig) -> None:
    (data_train, data_test, label_set_train, bilu_mappings) = load_and_prepare(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    trainset = TraingDataset(
        data=data_train.preprocessed, tokenizer=tokenizer,
        label_set=label_set_train, tokens_per_batch=cfg.hyperparams.tokens_per_batch
    )
    testset = TraingDataset(
        data=data_test.preprocessed, tokenizer=tokenizer,
        label_set=label_set_train, tokens_per_batch=cfg.hyperparams.tokens_per_batch
    )

    num_labels = len(trainset.label_set.ids_to_label.values())
    model_pretrained = BertForTokenClassification.from_pretrained(cfg.model.name,
        num_labels=num_labels).to(DEVICE)
    model_pretrained.config.id2label = label_set_train["ids_to_label"]
    model_pretrained.config.label2id = label_set_train["labels_to_id"]
    optimizer = AdamW(model_pretrained.parameters(), lr=cfg.hyperparams.learning_rate)

    trainloader = DataLoader(
        trainset,
        collate_fn=TraingingBatch,
        batch_size=cfg.hyperparams.batch_size,
        shuffle=cfg.hyperparams.shuffle
    )

    testloader = DataLoader(
        testset,
        collate_fn=TraingingBatch,
        batch_size=cfg.hyperparams.batch_size,
        shuffle=cfg.hyperparams.shuffle
    )

    # Modeling
    model_finetuned = train(cfg=cfg, model=model_pretrained, dataloader=trainloader, labelset=label_set_train, optimizer=optimizer,
        bilu_mappings=bilu_mappings, save_directory=cfg.caching.finetuned_ner_model)

    # Evaluation
    evaluate(cfg=cfg, model=model_finetuned, dataloader=testloader, labelset=label_set_train, bilu_mappings=bilu_mappings,
        save_directory=cfg.caching.finetuned_ner_metrics)

if __name__ == "__main__":
    main()

#TODO: extract predictions and add to output file incl. input incl/exl. criteria
