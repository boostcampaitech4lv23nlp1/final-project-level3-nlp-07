from transformers import AutoConfig, BartForConditionalGeneration, PreTrainedTokenizerFast
from arguments import cfg, args

# load config, tokenizer, model
def load_model_tokenizer():
    
    config = AutoConfig.from_pretrained(cfg.model.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.model.model_name_or_path, use_fast=False, cache_dir=args.cache_dir)
    model = BartForConditionalGeneration.from_pretrained(cfg.model.model_name_or_path, config=config, cache_dir=args.cache_dir)
    
    return model, tokenizer