import argparse
from dataclasses import dataclass, field
from omegaconf import OmegaConf

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config')
args, _ = parser.parse_known_args()
cfg = OmegaConf.load(f'./config/{args.config}.yaml')

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "huggingface에서 가져올 PLM"}) 
 
    use_slow_tokenizer: bool = field(
        default=True, 
        metadata={"help": "pass하면, slow tokenizer 사용함.(huggingface tokenizers library에서 지원되지 않음)"}) 


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, 
        metadata={"help": "사용할 dataset name(huggingface datasets 이름)"})

    dataset_config_name: str = field(
        default=None, 
        metadata={"help": "사용할 dataset의 config name(huggingface datasets의 config name)"})   

    train_file: str = field(
        default=None, 
        metadata={"help": "train data가 존재하는 csv 또는 json 파일"})  

    validation_file: str = field(
        default=None, 
        metadata={"help": "valid data가 존재하는 csv 또는 json 파일"})  
    
    overwrite_cache: bool = field(
        default=True, 
        metadata={"help": "cached train, eval set 덮어쓰기"}) 

    text_column: str = field(
        default=None, 
        metadata={"help": "전체 text를 포함하는 dataset의 column name(요약용)"}) 

    summary_column: str = field(
        default=None, 
        metadata={"help": "요약을 포함하는 dataset의 comumn name(요약용)"})

    output_dir: str = field(
        default=None, 
        metadata={"help": "최종 모델을 저장할 위치"}) 

    checkpointing_steps: str = field(
        default=None, 
        metadata={"help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."}) 

    resume_from_checkpoint: str = field(
        default=None, 
        metadata={"help": "If the training should continue from a checkpoint folder."}) 


@dataclass
class TrainArguments:
    ignore_pad_token_for_loss: bool = field(
        default=True, 
        metadata={"help": "loss 계산에서 padded label에 해당하는 token을 무시할지의 여부"})  

    max_source_length: int = field(
        default=1024, 
        metadata={"help": "최대 input sequence 길이. 이 값보다 긴 tokenized sequence는 잘리고, 짧은 sequence는 padding된다."})  

    source_prefix: str = field(
        default=None, 
        metadata={"help": "모든 source text 앞에 추가할 접두사(T5 모델에 유용)"})  

    preprocessing_num_workers: int = field(
        default=None, 
        metadata={"help": "전처리에 사용할 process 수"})  
 
    max_target_length: int = field(
        default=128, 
        metadata={"help": "tokenization 이후 target text의 최대 총 sequence length. 이 값보다 긴 sequence는 잘리고 짧은 sequence는 padding된다."})  

    val_max_target_length: int = field(
        default=None, 
        metadata={"help": "tokenization 이후 valid target text의 최대 총 sequence length. 이 값보다 긴 sequence는 잘리고 짧은 sequence는 padding된다."
        "기본 값은 'max_target_length'이다. 이 arg에 값을 주면, 'eval', 'predict'에서 사용되는 'model.generate'의 'max_length'를 재정의하는데 사용된다."})

    max_length: int = field(
        default=128, 
        metadata={"help": "tokenization 이후 최대 총 input sequence length. 이 값보다 긴 sequence는 잘리고, 'pad_to_max_length'가 True면 짧은 sequence는 padding된다."}) 

    num_beams: int = field(
        default=None, 
        metadata={"help": "eval에서 사용할 beam의 개수. 이 인수는 'eval', 'predict' 중에서 'model.generate'로 전달된다."}) 

    pad_to_max_length: str = field(
        default=True, 
        metadata={"help": "True로 설정하면, 모든 sample들을 'max_length'로 padding한다. dynamic padding이 사용된다."}) 

    per_device_train_batch_size: int = field(
        default=8, 
        metadata={"help": "training dataloader의 batch size(device 당)"}) 

    per_device_eval_batch_size: int = field(
        default=8, 
        metadata={"help": "evaluation dataloader의 batch size(device 당)"}) 
    
    learning_rate: float = field(
        default=5e-5, 
        metadata={"help": "사용할 초기 learning rate(potential warmup period 이후)"}) 

    weight_decay: float = field(
        default=0.0, 
        metadata={"help": "사용할 weight decay"}) 

    num_train_epochs: int = field(
        default=3, 
        metadata={"help": "train할 때의 총 epoch 수"}) 
    
    max_train_steps: int = field(
        default=None, 
        metadata={"help": "train할 때의 총 epoch 수. 이 곳에 값을 넣는다면, num_train_epochs를 이 값으로 재정의한다."}) 

    gradient_accumulation_steps: int = field(
        default=1, 
        metadata={"help": "backward/update padd를 수행하기 전, accumulate할 update step 수."}) 

    lr_scheduler_type: str = field(
        default="linear", 
        metadata={"help": "사용할 scheduler type"}) 
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]

    num_warmup_steps: int = field(
        default=0, 
        metadata={"help": "lr scheduler에서 warmup step 수"}) 
    
    seed: int = field(
        default=None, 
        metadata={"help": "reproducible training을 위한 seed"}) 

    model_type: str = field(
        default=None, 
        metadata={"help": "reproducible training을 위한 seed"}) 

    report_twith_trackingo: bool = field(
        default=True, 
        metadata={"help": "Whether to enable experiment trackers for logging."}) 

    report_to: str = field(
        default=all, 
        metadata={"help": 'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."}) 


@dataclass
class huggingfaceArguments:
    push_to_hub: bool = field(
        default=True, 
        metadata={"help": "huggingface hub에 model을 push할지의 여부"}) 

    hub_model_id: str = field(
        default=None, 
        metadata={"help": "`output_dir`과 동기화할 repo name"}) 
    
    hub_token: str = field(
        default=None, 
        metadata={"help": "model hub로 push하는데 사용할 토큰"}) 