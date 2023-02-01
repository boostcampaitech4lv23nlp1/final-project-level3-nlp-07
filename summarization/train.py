from data_loader import load_and_concat_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from check import check
from arguments import train_args
from utils import detect_last_checkpoint, set_seed
from arguments import cfg, args, train_args
from model import load_model_tokenizer
from logger import set_logging
from simcls_model import CandidateScorer
from simcls_trainer import Trainer, TrainConfig
import argparse
import torch


def train():
    def parse_args_to_config(args: argparse.Namespace) -> TrainConfig:
        return TrainConfig(lr=args.lr, 
                        batch_size=args.batch_size,
                        num_epochs=args.num_epochs, 
                        save_dir=args.save_dir,
                        weight_decay=args.weight_decay, 
                        margin_lambda=args.margin_lambda, 
                        eval_steps=args.eval_steps,
                        early_stopping_patience=-1)

    # check
    check()

    # logger
    logger = set_logging('train')

    # set seed
    set_seed(cfg.train.seed)
    
    # load dataset
    raw_datasets = load_and_concat_dataset(cfg.data.finetuning_dataset)

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

    trainer = Trainer(
        model=model,
        generator_path = cfg.model.model_name_or_path,
        roberta_path = cfg.model.roberta_path
    )

    # config = parse_args_to_config(cfg)
    config = parse_args_to_config(cfg.arg)

    # Training
    if train_args.args.do_train:

        train_result = trainer.train(train_dataset, eval_dataset, config)

if __name__ == "__main__":
    train()