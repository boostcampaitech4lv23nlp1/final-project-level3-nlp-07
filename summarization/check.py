from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
import nltk
from filelock import FileLock
from arguments import cfg

def check():
    # transformers의 최소 버전
    check_min_version("4.25.0")

    # 요구하는 datasets 버전
    require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

    # nltk가 제대로 다운로드 되었는지 확인
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)

    # Sanity checks
    if cfg.data.finetuning_dataset is None:
        raise ValueError("Need either a dataset name or a training/validation file.")

    if cfg.huggingface.push_to_hub:
        assert cfg.data.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."