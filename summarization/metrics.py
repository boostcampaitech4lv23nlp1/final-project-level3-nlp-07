# 학습 중에 모델을 평가할 수 있도록 trainer에게 compute_metrics() 함수 제공하기
# 요약 task -> ROUGE 점수를 계산하기 전에, pred와 label을 text로 decode 해야 한다.

from arguments import args
from model import load_model_tokenizer
from process_text import postprocess_text
import evaluate
import numpy as np


# rouge metric 불러오기
metric = evaluate.load("rouge")

model, tokenizer = load_model_tokenizer()


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # 모델이 예측 logit외에 다른 것을 리턴하는 경우
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # 모델 출력에서 평가지표가 사용할 수 있는 텍스트로 변환하기 위해서 tokenizer.batch_decode() 사용(생성된 요약을 텍스트로 decode)
    # ex -> decoded_preds : 엄마가 아빠 못하니까 그러지
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    if args.ignore_pad_token_for_loss:
        # 레이블 내의 -100을 pad_token_id(3)로 교체
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # passage 요약을 텍스트로 디코딩
    # ex -> decoded_labels : 기분 좋을 때는 아빠 급하거나 울 때는 엄마라고 한다.
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 단순 후처리
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # ROUGE score 계산
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    # ex -> result = {'rouge1': 0.77, 'rouge2': 0.03, 'rougeL': 0.75, 'rougeLsum': 0.75, 'gen_len': 18.14}

    return result