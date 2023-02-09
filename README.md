# You Only Use a Minute
> 카카오톡 오픈 채팅방의 대화 내용을 요약해주는 서비스
&nbsp;

> 최종 발표 [PDF](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07/blob/master/assets/%5B%EC%B5%9C%EC%A2%85%5DNLP_07_%EC%98%A4%ED%94%88%EC%B1%84%ED%8C%85%EB%B0%A9%EC%9A%94%EC%95%BD.pdf) & [YOUTUBE](https://www.youtube.com/watch?v=VgHl_u3EKKQ)

## 개요
**YOUM**은 카카오톡 오픈 채팅방의 대화 내용을 요약해주는 서비스입니다. 현재 카카오톡 오픈 채팅방은 월간 900만 ~ 1000만 명이 사용할 정도로 수요가 많은 서비스이며 그 수요는 꾸준히 증가하고 있습니다. 하지만 대화 참여자가 많아짐에 따라 대화의 흐름을 파악하기 어렵게 되고 원하는 주제가 아닌 대화가 오고 가게 되어 그에 대한 관심과 참여가 떨어지는 문제가 발생하고 있습니다.

**YOUM**은 오픈 채팅방 대화 요약으로 비가독성 문제를 해결하고자 합니다. **YOUM**을 통해 유저의 불편을 해소하고 유저 편의성을 증대하여 서비스의 실질 수요를 높일 수 있을 것이라 기대하고 있습니다.

## Architecture
<img src="https://user-images.githubusercontent.com/97818356/217647222-777821a2-311f-4419-8588-4e054729b499.PNG">

&nbsp;

## Members
김한성|염성현|이재욱|최동민|홍인희|
:-:|:-:|:-:|:-:|:-:
<img src='https://user-images.githubusercontent.com/44632158/208237676-ae158236-16a5-4436-9a81-8e0727fe6412.jpeg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/44632158/208237686-c66a4f96-1be0-41e2-9fbf-3bf738796c1b.jpeg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/108864803/208801820-5b050001-77ed-4714-acd2-3ad42c889ff2.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/108864803/208802208-0e227130-6fe5-4ca0-9226-46d2b07df9bf.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/97818356/217166687-a8ab141e-94e8-4a44-b976-ac801ff246ea.jpg' height=80 width=80px></img>|
[Github](https://github.com/datakim1201)|[Github](https://github.com/neulvo)|[Github](https://github.com/JaeUk2)|[Github](https://github.com/unknownburphy)|[Github](https://github.com/inni-iii)|
Project Design<br>DTS modeling<br>FrontEnd<br>Deployment|Dataset Search<br>FrontEnd<br>BackEnd<br>TroubleShooting|Data post-processing<br>Data Guideline<br>Summarization<br>DB build|Dataset Search<br>Data pre-processing<br>DTS<br>BackEnd|Project Manager<br>Data Guideline<br>Summarization<br>FrontEnd
&nbsp;

## Demo
데모 영상 삽입

&nbsp;  

## Train Data
* Dialogue Topic Segmentation : [AI HUB 주제별 텍스트 일상 대화 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=543)
* summarization : [AI HUB 한국어 대화 요약](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=117) 

&nbsp; 

## Model
### Dialogue Topic Segmentation
* 개요
    * 여러 대화를 주제적인 일관성을 가지는 segment로 나누는 방법
* 모델 구조
    * klue/roberta-large
    * 추가로 Coherence Scoring model이라는 이름의 MLP Head
* Train Dataset
    * 임의의 Dialogue로 부터 dialogue flow를 활용해 positive, negative utterance pair 생성
* finetuning
    * 생성된 positive pair, negative pair를 활용해 Marginal ranking loss로 모델 학습.

<img src="https://user-images.githubusercontent.com/97818356/217662439-00516d07-a329-48a3-a960-4ec43c9b964b.png" height=300 width=550px>

&nbsp; 

### Summarization
* 개요
    * 대화문을 토대로 내용을 파악하여 관련된 요약문을 생성
* 모델 구조
    * yeombora/kobart_r3f
    * yeombora/Kobart_SimCLS

<img src="https://user-images.githubusercontent.com/97818356/217663515-9e346fe6-7062-4b59-934f-2b23e0a0de01.PNG" height=300 width=550px>

&nbsp; 

## How to Run
* 빠른 실행은 http://49.50.162.64:30001/examples/dashboard.html 입니다. (업데이트 예정)

### Poetry dependency
```
$ poetry shell
$ poetry install
```
### Dialogue Topic Segmentation model download
```
$ git clone https://github.com/dataKim1201/dialouge_Topic_Segmentation
```

### bentoml packing
* app 폴더에 들어가서 실행하기
* DTS model path 변경 필요
```
$ python bentoml_packing.py
```
### server run
```
$ bentoml serve SummaryService:latest
```
* 다른 터미널에서 실행 (default port == 30001)
```
$ python backend_hub.py
```
### streamlit run
``` 
$ streamlit run app.py --server.port 30002
```

&nbsp; 

## 프로젝트 구조
``` 
├─ DTS
│  ├─ __init__.py
│  ├─ config
│  │  └─ config.yaml
│  ├─ ...
│  └─ main.py
│
├─ app
│  ├─ app.py
│  ├─ backend_hub.py
│  ├─ bentoml_packing.py
│  ├─ load_dataset.py
│  ├─ model.py
│  ├─ paper-dashboard-master
│  │  ├─ assets
│  │  │  ├─ js
│  │  │  │  ├─ core
│  │  │  │  │  ├─ bootstrap.min.js
│  │  │  │  │  ├─ jquery.min.js
│  │  │  │  │  └─ popper.min.js
│  │  │  │  ├─ ...
│  │  │  │  ├─ get_keyword.js
│  │  │  │  ├─ get_timeline.js
│  │  │  │  ├─ dashboard_input.js
│  │  │  │  └─ paper-dashboard.js
│  │  │  └─ scss
│  │  ├─ examples
│  ├─ prediction.py
│  ├─ requirements.txt
│  └─ validation.py
├─ deploy_ssh.sh
├─ notebooks
│  ├─ ...
│  ├─ extract_dialogue.ipynb
│  └─ predict_test_DTS.ipynb
├─ summarization
│  ├─ ...
│  └─ train.py
└─ utils
│   ├─ __init__.py
│   ├─ ...
│   └─ stopword.txt
├─ README.md
├─ poetry.lock
├─ py_template
├─ pyproject.toml
│
└── thanks for comming I'm Yeombora
```
## reference
* TextTiling: segmenting text into multi-paragraph subtopic passages
* Unsupervised Topic Segmentation of Meetings with BERT Embeddings
* Improving Unsupervised Dialogue Topic Segmentation with Utterance-Pair Coherence Scoring (Xing et al, 2021)
* BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
* SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization
* https://github.com/Hanul/BadWordFilter
