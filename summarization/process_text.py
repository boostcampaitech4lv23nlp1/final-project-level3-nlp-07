from arguments import args, train_args
from model import load_model_tokenizer
from logger import set_logging
import nltk

# T5 모델 사용할 때 설정해주기
prefix = args.source_prefix if args.source_prefix is not None else ""

# model, tokenizer 불러오기
model, tokenizer = load_model_tokenizer()

# padding 설정
padding = "max_length" if args.pad_to_max_length else False


# 전처리 진행
def preprocess_function(dataset):
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    # dialogue와 summary 가져오기
    inputs, targets = [], []
    for i in range(len(dataset['dialogue'])):
        if dataset['dialogue'][i] and dataset['summary'][i]:
            inputs.append(dataset['dialogue'][i])
            targets.append(dataset['summary'][i])
    inputs = [bos + inp + eos for inp in inputs]

    # tokenizing
    model_inputs = tokenizer(inputs, max_length=args.max_target_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=targets, max_length=args.max_target_length, padding=padding, truncation=True)

    # pad_token_id를 -100으로 변환 -> 패딩 토큰이 손실함수에 의해 무시되는지 확실화
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 후처리 진행
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # preds, labels 각 문장 끝에 줄바꿈
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels