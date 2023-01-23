from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
)
from arguments import cfg, args

def load_model_tokenizer(logger):
    # load config, tokenizer, model
    config = AutoConfig.from_pretrained(cfg.model.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.model.model_name_or_path, use_fast=False, cache_dir=args.cache_dir)
    model = BartForConditionalGeneration.from_pretrained(cfg.model.model_name_or_path, config=config, cache_dir=args.cache_dir)

    # 처음부터 모델을 생성하는 경우 인덱스 오류를 방지하기 위해 필요한 경우에만 임베딩 크기를 조정한다.
    # 작은 vocab에서 더 작은 임베딩 크기를 원하는 경우 이 테스트를 제거한다.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < args.max_source_length
    ):
        if args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {args.max_source_length}."
            )
            model.resize_position_embeddings(args.max_source_length)
        elif args.resize_position_embeddings:
            model.resize_position_embeddings(args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )
    return model, tokenizer