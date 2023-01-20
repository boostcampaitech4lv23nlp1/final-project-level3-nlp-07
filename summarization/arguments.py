from dataclasses import dataclass, field
from omegaconf import OmegaConf
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from typing import Optional

import argparse


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config')
args, _ = parser.parse_known_args()
cfg = OmegaConf.load(f'./config/{args.config}.yaml')


@dataclass
class Arguments:
    model_name_or_path: str = field(
        default=cfg.model.model_name_or_path, 
        metadata={"help": "huggingface에서 가져올 PLM"}) 
 
    use_fast: bool = field(
        default=cfg.model.use_fast, 
        metadata={"help": "Fast tokenizer 사용"}) 

    finetuning_dataset: str = field(
        default=cfg.data.finetuning_dataset, 
        metadata={"help": "train data가 존재하는 csv 또는 json 파일"})  
    
    overwrite_cache: bool = field(
        default=cfg.data.overwrite_cache, 
        metadata={"help": "cached train, eval set 덮어쓰기"}) 

    resume_from_checkpoint: str = field(
        default=cfg.data.resume_from_checkpoint, 
        metadata={"help": "checkpoint가 존재하는 폴더. 여기서부터 학습을 재진행 할 수 있다."}) 

    ignore_pad_token_for_loss: bool = field(
        default=cfg.arg.ignore_pad_token_for_loss, 
        metadata={"help": "loss 계산에서 padded label에 해당하는 token을 무시할지의 여부"})  

    max_source_length: Optional[int] = field(
        default=cfg.arg.max_source_length, 
        metadata={"help": "최대 input sequence 길이. 이 값보다 긴 tokenized sequence는 잘리고, 짧은 sequence는 padding된다."})  

    source_prefix: Optional[str] = field(
        default=cfg.arg.source_prefix,  # "summarize: "
        metadata={"help": "모든 source text 앞에 추가할 접두사(T5 모델에 유용)"})  
    
    preprocessing_num_workers: Optional[int] = field(
        default=cfg.arg.preprocessing_num_workers, 
        metadata={"help": "전처리에 사용할 process 수"})  

    max_train_samples: Optional[int] = field(
        default=cfg.arg.max_train_samples, 
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."})  
        
    max_target_length: Optional[int] = field(
        default=cfg.arg.max_target_length, 
        metadata={"help": "tokenization 이후 target text의 최대 총 sequence length. 이 값보다 긴 sequence는 잘리고 짧은 sequence는 padding된다."})  

    val_max_target_length: Optional[int] = field(
        default=cfg.arg.val_max_target_length, 
        metadata={"help": "tokenization 이후 valid target text의 최대 총 sequence length. 이 값보다 긴 sequence는 잘리고 짧은 sequence는 padding된다."
        "기v본 값은 'max_target_length'이다. 이 arg에 값을 주면, 'eval', 'predict'에서 사용되는 'model.generate'의 'max_length'를 재정의하는데 사용된다."})
    
    max_eval_samples: Optional[int] = field(
        default=cfg.arg.max_eval_samples, 
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."})

    num_beams: Optional[int] = field(
        default=cfg.arg.num_beams, 
        metadata={"help": "eval에서 사용할 beam의 개수. 이 인수는 'eval', 'predict' 중에서 'model.generate'로 전달된다."}) 

    pad_to_max_length: bool = field(
        default=cfg.arg.pad_to_max_length, 
        metadata={"help": "True로 설정하면, 모든 sample들을 'max_length'로 padding한다. dynamic padding이 사용된다."}) 

    resize_position_embeddings: Optional[bool] = field(
        default=cfg.arg.resize_position_embeddings, 
        metadata={"help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."})     

    cache_dir: Optional[str] = field(
        default=cfg.data.cache_dir,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},)                    
    

@dataclass
class training_arg_class:
    args: str = field(
        default = Seq2SeqTrainingArguments(
            do_train=True,
            do_eval=True,            

            fp16=True,
            learning_rate=cfg.train.learning_rate,                                # 사용할 초기 learning rate(potential warmup period 이후)
            load_best_model_at_end=True,                    
            num_train_epochs=cfg.train.num_train_epochs,                               # train할 때의 총 epoch 수
            overwrite_output_dir=cfg.train.overwrite_output_dir, 
            output_dir = cfg.data.output_dir,
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,                    # training dataloader의 batch size(device 당)
            per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,                     # evaluation dataloader의 batch size(device 당)
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,                     # backward/update padd를 수행하기 전, accumulate할 update step 수
            predict_with_generate=True,                        # 생성 메트릭(ROUGE, BLEU)을 계산하기 위해 generate를 사용할지 여부
            label_smoothing_factor=cfg.train.label_smoothing_factor,

            logging_strategy="steps",
            logging_steps=cfg.train.logging_steps,
            evaluation_strategy="steps",
            eval_steps=cfg.train.eval_steps,            
            save_total_limit=3,
            save_strategy="steps",
            save_steps=cfg.train.save_steps,
            weight_decay=cfg.train.weight_decay,                      # 사용할 weight decay
            report_to=["wandb"],                          # 결과 및 로그를 보고할 통합 목록. 지원되는 플랫폼은 "azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard", "clearl", "wandb"       
            push_to_hub=cfg.huggingface.push_to_hub,                      # huggingface hub에 model을 push할지의 여부
            hub_private_repo=cfg.huggingface.hub_private_repo,                  # huggingface hub에 private로 설정할지 여부
            hub_token=cfg.huggingface.hub_token,                         # model hub로 push하는데 사용할 토큰                      
        )
    )

# parser
parser = HfArgumentParser(
        (Arguments, training_arg_class))
args, train_args = parser.parse_args_into_dataclasses()