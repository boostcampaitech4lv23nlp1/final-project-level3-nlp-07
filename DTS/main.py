import torch
import torch.nn as nn
from model import CSModel
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import OmegaConf
import wandb
import argparse
import torch.nn as nn
from load_dataset import *
from utils import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"
def train(cfg):
    ## Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model = CSModel(pretrained_id = cfg.model.model_name)
        
    model.parameters
    model.to(device)
    
    optimizer = optim.AdamW([
                {'params': model.plm.parameters()},
                {'params': model.cs_model.parameters(), 'lr': cfg.train.second_lr},
                    ], lr=cfg.train.lr,eps = 1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.T_0, T_mult=cfg.train.T_mult, eta_min=cfg.train.eta_min)
    optimizers = (optimizer,scheduler)
    
    ## load dataset 
    train_inputs = load_data(cfg.data.train_data)
    validation_inputs = load_data(cfg.data.valid_data)
    train_dataset = trainDataset(train_inputs,tokenizer)
    valid_dataset = trainDataset(validation_inputs,tokenizer)
    print(f'train_len : {len(train_dataset)}, valid_len : {len(valid_dataset)} now loading ...')
    data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)


    # model.plm.resize_token_embeddings(len(RE_train_dataset.tokenizer))

    
    
    ## train arguments
    training_args = TrainingArguments(
        output_dir=cfg.train.checkpoint,
        save_total_limit=5,
        save_steps=cfg.train.warmup_steps,
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        
        label_smoothing_factor = 0.1,
        
        per_device_train_batch_size=cfg.train.batch_size,    # default : 16
        per_device_eval_batch_size=cfg.train.batch_size,     # default : 16
        warmup_steps=cfg.train.warmup_steps,               
        weight_decay=cfg.train.weight_decay,               
    
        # for log
        logging_steps=cfg.train.logging_step,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.warmup_steps,                 # evaluation step.
        load_best_model_at_end = True,
        
        metric_for_best_model= 'eval_loss',
        greater_is_better=False,                             # False : loss 기준으로 최적화 해봄 도르
        dataloader_num_workers=cfg.train.num_workers,
        fp16=True,
        group_by_length = True,

        # push_to_hub=cfg.huggingface.push_to_hub,                      # huggingface hub에 model을 push할지의 여부
        # hub_private_repo=cfg.huggingface.hub_private_repo,                  # huggingface hub에 private로 설정할지 여부
        # hub_token=cfg.huggingface.hub_token,                         # model hub로 push하는데 사용할 토큰                      
        # push_to_hub_organization=cfg.huggingface.push_to_hub_organization,
        # hub_model_id =  cfg.huggingface.hub_model_id,
        # wandb
        report_to="wandb",
        run_name= cfg.wandb.exp_name
        )
    # data_collator = DataCollatorWithPadding(tokenizer,padding=True)
    trainer = MarginalTrainer(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        # data_collator = data_collator,
        train_dataset= train_dataset,  # training dataset
        eval_dataset= valid_dataset,     # evaluation dataset use dev
        compute_metrics=compute_metrics,  # define metrics function
        optimizers = optimizers
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    ## train model
    trainer.train()
    
    ## save model
    # model.save_model(cfg.model.saved_model)
    torch.save(model,cfg.model.saved_model)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    ## parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    ## set seed
    seed_everything(cfg.train.seed)
    wandb.login()
    wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)

    print('------------------- train start -------------------------')
    train(cfg)

    ## wandb finish
    wandb.finish()
    # dataloader 문제 확인