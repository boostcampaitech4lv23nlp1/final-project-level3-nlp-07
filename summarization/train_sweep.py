from data_loader import load_and_concat_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from check import check
from utils import detect_last_checkpoint, set_seed
from arguments import cfg, args
from model import load_model_tokenizer
from logger import set_logging
from process_text import preprocess_function
from metrics import compute_metrics

import torch
import wandb
import yaml


def train():
    # wandb
    wandb.init()
    sweep = wandb.config

    train_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=True,            

        fp16=True,
        learning_rate=cfg.train.learning_rate,                                # 사용할 초기 learning rate(potential warmup period 이후)
        load_best_model_at_end=True,                    
        num_train_epochs=sweep.epochs,                               # train할 때의 총 epoch 수
        overwrite_output_dir=cfg.train.overwrite_output_dir, 
        output_dir = cfg.data.output_dir,
        per_device_train_batch_size=sweep.batch_size,                    # training dataloader의 batch size(device 당)
        per_device_eval_batch_size=sweep.batch_size,                     # evaluation dataloader의 batch size(device 당)
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,                     # backward/update padd를 수행하기 전, accumulate할 update step 수
        predict_with_generate=True,                        # 생성 메트릭(ROUGE, BLEU)을 계산하기 위해 generate를 사용할지 여부
        label_smoothing_factor=sweep.label_smoothing_factor,

        logging_strategy="steps",
        logging_steps=cfg.train.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.train.eval_steps,            
        save_total_limit=3,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        weight_decay=sweep.weight_decay,                      # 사용할 weight decay
        report_to=["wandb"],                          # 결과 및 로그를 보고할 통합 목록. 지원되는 플랫폼은 "azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard", "clearl", "wandb"       
        push_to_hub=cfg.huggingface.push_to_hub,                      # huggingface hub에 model을 push할지의 여부
        hub_private_repo=cfg.huggingface.hub_private_repo,                  # huggingface hub에 private로 설정할지 여부
        hub_token=cfg.huggingface.hub_token,                         # model hub로 push하는데 사용할 토큰                      
    )
        

    # check
    check()

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # logger
    logger = set_logging('train')

    # import last_checkpoint
    last_checkpoint = detect_last_checkpoint(logger)

    # set seed
    set_seed(cfg.train.seed)
    
    # load dataset
    raw_datasets = load_and_concat_dataset(cfg.data.finetuning_dataset)
    print(raw_datasets)

    # load model & tokenizer
    model, tokenizer = load_model_tokenizer(logger)
    model.to(device)

    if train_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with train_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if train_args.do_eval:
        max_target_length = args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            args.max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        with train_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    label_pad_token_id  = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if train_args.fp16 else None,
    )
    
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=eval_dataset if train_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if train_args.predict_with_generate else None,
    )

    # Training
    if train_args.do_train:
        checkpoint = None
        if train_args.resume_from_checkpoint is not None:
            checkpoint = train_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            args.max_train_samples if args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        train_args.generation_max_length
        if train_args.generation_max_length is not None
        else args.val_max_target_length
    )
    num_beams = args.num_beams if args.num_beams is not None else train_args.generation_num_beams
    if train_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": cfg.model.model_name_or_path, "tasks": "summarization"}
    if train_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results

if __name__ == "__main__":
    ## yaml 파일 경로 설정
    with open('./config/sweep_config.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config, entity=cfg.wandb.entity) # project name 설정
    wandb.agent(sweep_id, function=train) # count = 몇 번 sweep 돌 것인지