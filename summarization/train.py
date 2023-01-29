from data_loader import load_and_concat_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from check import check
from arguments import train_args
from utils import detect_last_checkpoint, set_seed
from arguments import cfg, args, train_args
from model import load_model_tokenizer
from logger import set_logging
from process_text import preprocess_function
from metrics import compute_metrics
from simcls_model import CandidateScorer
from simcls_trainer import Trainer, TrainConfig
import argparse
import torch


def train():

    # def parse_args_to_config(args: argparse.Namespace) -> TrainConfig:
        # return TrainConfig(lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, save_dir=args.save_dir,
        #                 weight_decay=args.weight_decay, margin_lambda=args.margin_lambda, eval_steps=args.eval_steps,
        #                 early_stopping_patience=args.early_stop_patience)
    def parse_args_to_config() -> TrainConfig:
        return TrainConfig(lr=5e-5, batch_size=2, num_epochs=5, save_dir="/opt/ml/final-project-level3-nlp-07/summarization/simcls_model",
                        weight_decay=0.1, margin_lambda=0.01, eval_steps=1000,
                        early_stopping_patience=-1)

    # check
    # check()

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # logger
    # logger = set_logging('train')

    # import last_checkpoint
    # last_checkpoint = detect_last_checkpoint(logger)

    # set seed
    set_seed(cfg.train.seed)
    
    # load dataset
    raw_datasets = load_and_concat_dataset(cfg.data.finetuning_dataset)
    print(raw_datasets)

    # load model & tokenizer

    model = CandidateScorer(cfg.model.roberta_path)

    if train_args.args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))


    if train_args.args.do_eval:
        max_target_length = args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            args.max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))


    # Data collator
    label_pad_token_id  = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=8 if train_args.args.fp16 else None,
    # )
    


    # Initialize our Trainer
    """
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset if train_args.args.do_train else None,
        eval_dataset=eval_dataset if train_args.args.do_eval else None,
        args=train_args.args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if train_args.args.predict_with_generate else None,
    )
    """

    trainer = Trainer(
        model=model,
        generator_path = cfg.model.model_name_or_path,
        roberta_path = cfg.model.roberta_path
    )

    # config = parse_args_to_config(cfg)
    config = parse_args_to_config()

    # Training
    if train_args.args.do_train:

        train_result = trainer.train(train_dataset, eval_dataset, config)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        # metrics = train_result.metrics
        # max_train_samples = (
        #     args.max_train_samples if args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    results = {}
    # max_length = (
    #     train_args.args.generation_max_length
    #     if train_args.args.generation_max_length is not None
    #     else args.val_max_target_length
    # )
    # num_beams = args.num_beams if args.num_beams is not None else train_args.args.generation_num_beams
    # if train_args.args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    #     max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # kwargs = {"finetuned_from": cfg.model.model_name_or_path, "tasks": "summarization"}
    # if train_args.args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

    # return results

if __name__ == "__main__":
    train()