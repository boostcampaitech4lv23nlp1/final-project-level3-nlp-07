import json
import logging
import math
import os
import random
from pathlib import Path
import datasets
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from check import check
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import get_full_repo_name
from util import summarization_name_mapping, postprocess_text
from arguments import cfg
from dataclasses import dataclass, field
import wandb

logger = get_logger(__name__)

def main():
    # Reset the Memory
    torch.cuda.empty_cache()

    # 라이브러리 버전 체크
    check()

    # accelerator 초기화. with_tracking 설정 시, 초기화 해준다.
    accelerator_log_kwargs = {}

    if cfg.train.with_tracking:
        accelerator_log_kwargs["log_with"] = cfg.train.report_to
        accelerator_log_kwargs["logging_dir"] = cfg.data.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=cfg.train.gradient_accumulation_steps, **accelerator_log_kwargs)
    if cfg.train.source_prefix is None and cfg.model.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # 각 프로세스에 로그를 하나씩 생성합니다.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # set seed
    set_seed(cfg.train.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.huggingface.push_to_hub:
            if cfg.huggingface.hub_model_id is None:
                repo_name = get_full_repo_name(Path(cfg.data.output_dir).name, token=cfg.huggingface.hub_token)
            else:
                repo_name = cfg.huggingface.hub_model_id
            repo = Repository(cfg.data.output_dir, clone_from=repo_name)

            with open(os.path.join(cfg.data.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif cfg.data.output_dir is not None:
            os.makedirs(cfg.data.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # load dataset
    if cfg.data.dataset_name is not None:
        raw_datasets = load_dataset(cfg.data.dataset_name, cfg.data.dataset_config_name)
    else:
        data_files = {}
        if cfg.data.train_file is not None:
            data_files["train"] = cfg.data.train_file
        if cfg.data.validation_file is not None:
            data_files["validation"] = cfg.data.validation_file
        extension = cfg.data.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field='data')

    # download model & vocab.
    config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, use_fast=not cfg.model.use_slow_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model.model_name_or_path,
            from_tf=bool(".ckpt" in cfg.model.model_name_or_path),
            config=config,
    )

    # 처음부터 모델을 생성하는 경우 인덱스 오류를 방지하기 위해 필요한 경우에만 임베딩 크기를 조정한다.
    # 작은 vocab에서 더 작은 임베딩 크기를 원하는 경우 이 테스트를 제거한다.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = cfg.train.source_prefix if cfg.train.source_prefix is not None else ""

    # dataset을 전처리하는 과정이다. 먼저, 모든 text들을 tokenize한다.
    column_names = raw_datasets["train"].column_names

    # 입력/대상에 대한 column name을 가져온다.
    dataset_columns = summarization_name_mapping.get(cfg.data.dataset_name, None)
    if cfg.data.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = cfg.data.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{cfg.data.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if cfg.data.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = cfg.data.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{cfg.data.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    max_target_length = cfg.train.max_target_length
    padding = "max_length" if cfg.train.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=cfg.train.max_source_length, padding=padding, truncation=True)

        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and cfg.train.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.train.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    label_pad_token_id = -100 if cfg.train.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=cfg.train.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=cfg.train.per_device_eval_batch_size)

    # Optimizer
    # optimizer_grouped_parameters에서 하나는 weight decay가 있고, 나머지 하나는 weight decay가 없다.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.train.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.train.learning_rate)

    # train step에서의 Scheduler, math
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if cfg.train.max_train_steps is None:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=cfg.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.train.num_warmup_steps * cfg.train.gradient_accumulation_steps,
        num_training_steps=cfg.train.max_train_steps * cfg.train.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.train.num_train_epochs = math.ceil(cfg.train.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = cfg.data.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Metric
    metric = evaluate.load("rouge")

    # train
    total_batch_size = cfg.train.per_device_train_batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.train.max_train_steps}")

    progress_bar = tqdm(range(cfg.train.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # 이전 save에서의 weights, states load
    if cfg.data.resume_from_checkpoint:
        if cfg.data.resume_from_checkpoint is not None or cfg.data.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {cfg.data.resume_from_checkpoint}")
            accelerator.load_state(cfg.data.resume_from_checkpoint)
            path = os.path.basename(cfg.data.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # 수정된 날짜별로 폴더를 정렬한다. 가장 마지막이 최근의 checkpoint이다.
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, cfg.train.num_train_epochs):
        model.train()
        if cfg.train.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            if cfg.data.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                if cfg.train.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if cfg.data.output_dir is not None:
                        output_dir = os.path.join(cfg.data.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= cfg.train.max_train_steps:
                break

        model.eval()
        if cfg.train.val_max_target_length is None:
            cfg.train.val_max_target_length = cfg.train.max_target_length

        gen_kwargs = {
            "max_length": cfg.train.val_max_target_length if cfg is not None else config.max_length,
            "num_beams": cfg.train.num_beams,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not cfg.train.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if cfg.train.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        logger.info(result)

        if cfg.train.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

        if cfg.huggingface.push_to_hub and epoch < cfg.train.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                cfg.data.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(cfg.data.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if cfg.data.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if cfg.data.output_dir is not None:
                output_dir = os.path.join(cfg.data.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if cfg.data.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            cfg.data.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(cfg.data.output_dir)
            if cfg.huggingface.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(cfg.data.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)


if __name__ == "__main__":

    if cfg.wandb.wandb_mode:
        wandb.login()
        wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)
        main()
        wandb.finish()
    else:
        main()