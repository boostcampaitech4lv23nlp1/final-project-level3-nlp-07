from data_loader import load_and_concat_dataset, load_data
from datasets import load_metric
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)
from tokenize_data import preprocess_function
import torch
import os
from datasets import Dataset, DatasetDict, concatenate_datasets
import nltk
import numpy as np
from transformers.trainer_utils import get_last_checkpoint
import logging
import os


def train():
    logger = logging.getLogger(__name__)

    args = Seq2SeqTrainingArguments(
        output_dir = f"/opt/ml/input/summary_copy/test",
        #resume_from_checkpoint=None,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=50,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        do_train = True,
        do_eval=True,
        overwrite_output_dir=False, 
        logging_strategy = "steps",
        logging_steps = 50,
        eval_steps = 100,
        load_best_model_at_end = True,
        save_steps = 100,

    )
    source_prefix="summarize: "
    ## Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(42)
    
    # load dataset
    # train_datasets = load_data('/opt/ml/input/summarization_data/train/시사교육.json')
    # valid_datasets = load_data('/opt/ml/input/summarization_data/valid/시사교육.json')
    # raw_datasets = DatasetDict({'train':train_datasets, 'validation':valid_datasets})
    raw_datasets = load_and_concat_dataset('/opt/ml/input/summarization_data/')
    print(raw_datasets)

    # tokenizer, model
    config = AutoConfig.from_pretrained('gogamza/kobart-base-v2', use_auth_token=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2',use_fast=False, use_auth_token=True)
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2', config=config, use_auth_token=True)
    model.to(device)

    # 처음부터 모델을 생성하는 경우 인덱스 오류를 방지하기 위해 필요한 경우에만 임베딩 크기를 조정한다.
    # 작은 vocab에서 더 작은 임베딩 크기를 원하는 경우 이 테스트를 제거한다.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    max_source_length = 512
    resize_position_embeddings = None

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < max_source_length
    ):
        if resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {max_source_length}."
            )
            model.resize_position_embeddings(max_source_length)
        elif resize_position_embeddings:
            model.resize_position_embeddings(max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )
    source_prefix = ""
    prefix = source_prefix if source_prefix is not None else ""

    # Get the column names for input/target.
    text_column = 'dialogue'
    summary_column = 'summary'

    # Temporarily set max_target_length for training.
    max_target_length = 512
    padding = "max_length"
    label_smoothing_factor = 0.01

    def preprocess_function(dataset):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(dataset[text_column])):
            if dataset[text_column][i] and dataset[summary_column][i]:
                inputs.append(dataset[text_column][i])
                targets.append(dataset[summary_column][i])

        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    preprocessing_num_workers = None
    max_train_samples = None
    overwrite_cache = False

    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if max_train_samples is not None:
            max_train_samples = min(len(train_dataset), max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                # remove_columns=column_names,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    val_max_target_length = None
    max_eval_samples = None

    if args.do_eval:
        max_target_length = val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                #remove_columns=column_names,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    ignore_pad_token_for_loss = True
    label_pad_token_id  = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if args.fp16 else None,
    )
    import evaluate
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.predict_with_generate else None,
    )

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            max_train_samples if max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    num_beams = None

    # Evaluation
    results = {}
    max_length = (
        args.generation_max_length
        if args.generation_max_length is not None
        else val_max_target_length
    )
    num_beams = num_beams if num_beams is not None else args.generation_num_beams
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = max_eval_samples if max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": 'gogamza/kobart-base-v2', "tasks": "summarization"}
    if args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    train()


if __name__ == "__main__":
    train()