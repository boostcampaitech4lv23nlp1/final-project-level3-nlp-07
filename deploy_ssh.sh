#!/bin/bash
git pull origin main               # 최신 정보를 가져와서
#poetry로 한다면?
poetry shell
pip install -r requirements.txt    # 사실 이것도 안해도됨
python3 nltk.txt