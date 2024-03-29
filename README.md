# Relation Extraction Competition
> Boostcamp AI Tech 5기 Level 2 죠죠의 기묘한 모험

<br>

## Leader Board
🥇 **Private 1st**
![lb](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/073aa3a9-c997-4e9f-b63a-5ea6dde13053)

<br>

## 개요

단어 간의 관계를 파악하는 것은 문장의 의미나 의도를 해석하는 데 많은 도움을 준다. 관계 추출(Relation Extraction)은 단어(Entity) 간의 관계를 예측하는 문제이다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요하다. 문장 및 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델의 성능을 높이는 것이 이번 프로젝트의 목표이다.

<br>

## 멤버 & 역할

### 멤버
|전민수|조민우|조재관|진정민|홍지호|
|:-:|:-:|:-:|:-:|:-:|
|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/e1fd55d4-617a-436e-9ab0-e18eaeda685c' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/1060e554-e822-4bac-9d7e-ddafdbf7d9c1' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/5038030e-b30c-43e1-a930-3c63a1332843' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/f871e7ea-7b41-494d-a858-2e6b2df815b9' height=125 width=125></img>|<img src='https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/102800474/f5914167-bf44-40b6-8c78-964a8fb90b10' height=125 width=125></img>|
|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/line1029)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/Minwoo0206)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/jaekwanyda)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/wjdals3406)|[<img src='https://img.shields.io/badge/GitHub-181717?style&logo=github&logoColor=white' ></img>](https://github.com/jiho-hong)|

### 역할

| 이름   | 역할                                                         |
| ------ | ------------------------------------------------------------ |
| 전민수 | 베이스코드 재작성, 모델 설계, 논문 구현, 하이퍼파라미터 서칭 |
| 조민우 | 모델 설계 및 실험, 하이퍼파라미터 서칭                       |
| 조재관 | 데이터 증강 및 분석, 모델 탐색, 오답 분석                    |
| 진정민 | 모델 설계 및 실험, 하이퍼파라미터 서칭                       |
| 홍지호 | 모델 설계 및 실험, 데이터 증강 및 분석                                       |

<br>

## 협업

### Meeting

- 매주 월요일마다 오프라인으로 모여 주간 계획 수립 및 학습 정보 공유
- 매일 아침 10시에 데일리 스크럼을 진행하며 실험 결과 및 학습 계획 공유
- 매일 오후 4시에 피어세션을 진행하며 실험 결과 분석 및 향후 실험 계획 수립

### 협업툴

- Notion
- Git
- W&B

<br>

## Skill

- Pytorch
- HuggingFace
- Pandas

<br>

## Directory
```
  level2_klue-nlp-11   
  ├── README.md   
  ├── config   
  │   ├── config.py   
  │   ├── config.yaml   
  │   └── sweep_config.yaml   
  ├── dataloader.py   
  ├── inference.py   
  ├── models.py   
  ├── pretraining.py   
  ├── sweep.py   
  ├── train.py   
  └── utils   
      ├── dict_label_to_num.pkl      
      ├── dict_num_to_label.pkl   
      ├── losses.py   
      ├── metrics.py   
      ├── seed.py   
      └── utils.py   
```

<br>

## EDA

관계 추출 프로젝트를 위해 제공된 데이터는 다음과 같습니다. 

이전 프로젝트와 달리 validation dataset은 따로 제공되지 않았습니다.

- train.csv: 총 32,470개
- test_data.csv: 총 7,765개
- 데이터 예시
   
   ![Untitled](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/c2ed7146-4f9f-47ca-bb69-1fecfb8b15c8)
 

### 데이터 분포

아래 데이터 분포를 살펴보면 데이터 분포에 대한 두 가지 사실을 알 수 있습니다.

1. label에 따른 데이터 개수가 불균형하다. 
2. source마다 label 분포가 불균형하다.

<details>
<summary>전체 데이터 분포</summary>

![Untitled 1](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/a48ea5fe-731a-4851-b3ec-98be6413917b)

</details>

<details>
<summary>소스별 데이터 분포</summary>

<details>
<summary>source = ‘wikitree’</summary>

![Untitled 2](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/fa8e5690-bcb2-4179-8337-97852bdb5e87)

</details>

<details>
<summary>source = ‘wikipedia’</summary>

![Untitled 3](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/60e697d3-8aad-4a4d-90ee-3b079349ea11)

</details>

<details>
<summary>source = ‘policy_briefing’</summary>

![Untitled 4](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/0fb84fa1-f676-4286-b5df-6c6c6660a984)

</details>

</details>


<br>

## Data Experiments

### Data Split

validation dataset이 제공되지 않았기 때문에 자체적으로 validation dataset을 구축해야 했습니다. data split의 비율은 경험적 근거에 따라서 10%로 정했지만, sentence는 같고 subject entity와 object entity는 다른 데이터를 어떻게 split해야 할지 고민했습니다. 

- 문제 데이터 예시

    ![Untitled 5](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/9a36f254-8767-4229-a1d9-4a08e6d525cc)
    

해당 데이터가 train과 validation dataset에 각각 들어가게 되면 과적합 가능성이 있다고 판단하였습니다. 이에 따라 중복 데이터를 허용하지 않는 test dataset을 일시적으로 구축한 후 다음과 같은 두 가지 경우에 val_f1_score와 test_f1_score의 차이값을 비교하였습니다.

1. sentence가 같은 경우라도 무작위로 train 혹은 validation dataset으로 split
2. sentnece가 같은 경우에는 train 혹은 validation dataset 한쪽으로만 split

실험 결과 2번이 val_f1_score와 test_f1_score의 차이가 더 작았습니다. 따라서 validation dataset은 전체 train dataset의 10% 비율로 추출하되 문제 데이터의 경우 train 혹은 validation dataset 한쪽에 몰아넣기로 했습니다.

- 실험 결과
    
    value = |val_f1_score - test_f1_score|
    
    |              | avg      | seed a   | seed b   | seed c   |
    | ------------ | -------- | -------- | -------- | -------- |
    | case 1(양쪽) | 0.869156 | 1.15831  | 0.554001 | 0.895157 |
    | case 2(한쪽) | 0.163386 | 0.203056 | 0.067093 | 0.22001  |

### Typed Entity Marker

관계 추출에 관한 논문인 [Matching the Blanks](https://aclanthology.org/P19-1279.pdf)와 [An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/pdf/2102.01373.pdf)을 읽고 해당 내용을 적용하였습니다.

A. Matching the Blanks

![Untitled 6](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/c48f8229-97bc-4fd8-9332-f2d90f3cf15a)

위 사진 기준으로 [E1] token과 [E2] token을 추가한 뒤 [CLS] token 정보와 함께 concatenate하여 classifie로 전달하도록 구현했습니다. 다음과 같은 세 가지 방법을 비교하였습니다.

1. [CLS] token 사용
2. [CLS], [E1]-앞, [E1]-뒤, [E2]-앞, [E2]-뒤 token 사용
3. [CLS], [E1]-앞, [E2]-앞 token 사용

실험 결과 3번이 가장 성능이 좋았습니다. 

- **LB Score(제출 점수): 64 → 70.43**

B. An Improve Baseline for Sentence-level Relation Extraction

[E1] 혹은 [E2] token을 다음과 같은 두 가지 방법으로 설정하였습니다.

1. 이미 주어진 entity type에 대한 정보를 special token으로 새롭게 추가합니다.
예시: <S: PERSON> Bill was born in <O:CITY> Seattle. ← Typed entity marker
2. 이미 tokenizer에 포함되었지만, corpus에서 한 번도 사용되지 않은 토큰을 사용합니다.
예시: @ * person * Bill @  was born in # ^ city ^ Seattle #.  ← Typed entity marker(punct)

실험 결과 2번이 가장 성능이 높았습니다. 이는 이미 학습된 토큰이 처음부터 학습을 다시 해야 하는 토큰보다 더 효과적이기 때문이라고 추측할 수 있습니다.

- **LB Score(제출 점수): 70.43 → 71.028**

### Semantic Typing

Bert와 같은 사전 학습 모델이 학습하는 방식에 Next Sentence Prediction이 있는 것에 착안하여 원문장의 앞부분에 자연어 형태로 entity 정보를 넣어주면 모델 성능이 올라갈 것으로 생각했고 해당 내용이 구현된 논문([Unified Semantic Typing with Meaningful Label Inference](https://arxiv.org/pdf/2205.01826v1.pdf))을 바탕으로 여러 종류의 쿼리를 구성해 모델을 실험했습니다.

- 사용한 쿼리
    
    1) [Subject]와 [Object] 사이의 관계는 무엇인가?
    
    2) [Subject]와 [Object]의 관계는 [Subject:type]와 [Object:type]의 관계이다.
    
    3) [Object]는 [Subject:type]인 [Subject]의 [Object:type]이다.
    
- 각 쿼리를 자연어 형태로 구성한 버전과 Typed entity marker(punct)를 적용한 형태로 구성한 버전을 각각 실험했습니다.
    
    예1: Bill과 Seattle 사이의 관계는 무엇인가? + sentence2
    
    예2: @ * person * Bill과 # ^ city ^ Seattle 사이의 관계는 무엇인가? + sentence2
    

실험 결과 2번 형태의 쿼리를 자연어 형태로 표현했을 때 가장 성능이 좋았습니다.

- **LB Score(제출 점수): 71.028 → 74.2119**

### Confusion Matrix

실험을 진행하면서 나오는 오답의 분포를 관찰하기 위해서 Confusion Matrix를 만들었습니다. Confusion Matrix를 만들기 전에는 학습 데이터가 부족한 label의 오답 비율이 높을 것으로 생각했습니다. 그러나 예상과는 다르게 학습 데이터가 가장 많은 no_relation의 오답 비율이 꽤 높음을 확인할 수 있었습니다.

- 결과
    
    ![image](https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/0f5c6a23-9e0a-4a68-99c1-60927c0ceb96)
    

### Data Augmentation

no_relation의 오답 비율이 높으므로 데이터가 적은 label에 대해서만 데이터 증강을 적용하려는 원래의 계획을 모든 label에 대해서 데이터 증강을 적용하는 것으로 수정했습니다. 번역을 두 번 통과하더라도 단어 사이의 관계는 변하지 않을 것이라는 가정하에 backtranslation 기법을 적용하였습니다. 그 결과 LB Score의 상승을 꾀할 수 있었습니다.

- **LB Score(제출 점수) : 72.052 → 73.8259**

### Source Token

data source(wikitree, wikipedia, policy_briefing)에 따른 label 분포가 다르므로 해당 정보를 token 형태로 제공하면 모델이 label을 맞추는 데 도움이 될 것으로 생각했습니다. 이에 따라 [WT], [WP], [PB] 토큰 중 하나를 Semantic Typing 문장 앞 혹은 원문장의 앞에 추가했습니다.

- 예1: [CLS] [WT] sentence1 [SEP] sentence2
- 예2: [CLS] sentence1 [SEP] [WT] sentence2

그러나 결과적으로 두 경우 모두 source token을 추가하지 않은 모델과 성능 차이가 거의 없었습니다. 이는 label 분포가 wikitree와 wikipedia와 확연히 다른 policy_briefing의 데이터가 얼마 없었기 때문으로 추측됩니다.

<br>

## Loss Function

데이터 불균형 문제를 해결하기 위해서 기본적으로 설정된 Cross Entropy Loss 이외에 Class-Balanced Loss와 Focal Loss를 활용하였습니다.

### CE Loss

- 다중 클래스 분류에 주로 사용되는 loss입니다.
- baseline에 설정되어 있던 loss로, 기본 loss로 설정하였습니다.

### CB Loss

- 클래스 불균형 문제를 해결하기 위해서 효과적인 표본 수가 높을수록 낮은 가중치를 부여합니다.
- 참고 논문 - [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/pdf/1901.05555.pdf)

### Focal Loss

- 클래스 불균형 문제를 해결하기 위해서 잘못 분류된 클래스나 어려운 클래스에 더 높은 가중치를 부여합니다.
- 참고 논문 - [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)

Focal Loss를 적용했을 때 모델 성능이 가장 좋았습니다.

- **LB Score(제출 점수) : 73.8014 → 74.0923**

<br>

## Modeling

### TAPT

[Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/pdf/2004.10964.pdf)를 읽고 해당 내용을 적용해 보았습니다. 지금까지는 huggingface에서 pretrained-model을 불러와서 downstream-task에 대해 fine-tuning을 진행했지만, 이번에는 저희 task에 specific하게 pretrained된 모델을 구축하고자 했습니다. 이에 따라 huggingface에서 pretrained-model을 불러온 후 저희 데이터에 대해서 MLM(Masked Language Model) 기법으로 재학습시켜 주었습니다.

그 결과 기존 모델과 비교했을 때 성능 차이가 거의 나지 않았습니다. re-pretraining을 위해 사용한 데이터가 wikitree, wikipedia, policy_briefing에서 추출한 데이터인데 wikitree와 wikipedia의 데이터는 사실 기존에 pretraining을 위해 사용된 데이터와 크게 다르지 않을 거라는 생각이 들었습니다. 또한, task-specific 하게 re-pretraining하기 위해서는 subject와 object만 높은 확률로 masking하는 방식을 사용했어야 했다는 생각도 들었지만 시간 문제로 해당 방법을 적용해 보지는 못했습니다.

  

### LSTM

문장의 일부 토큰의 결과 벡터만 이용했던 것에서 문장 전체 벡터를 이용할 수 있는 RNN구조로 변경해 보았습니다. KLUE-RoBERTa와 3개의 linear layer로 이루어진 기존 모델에 LSTM, Bi-LSTM, GRU를 추가하여 실험하였습니다.

<img width="1355" alt="Untitled 8" src="https://github.com/boostcampaitech5/level2_klue-nlp-11/assets/74582277/dfaa69ba-5e71-4cce-8592-a076dd786efe">

그러나 기존 모델보다 향상된 결과를 얻을 수 없었습니다.

- 결과
    
    
    |         | validation f1 | train loss | validation loss |
    | ------- | ------------- | ---------- | --------------- |
    | base    | 85.852        | 0.4184     | 0.5429          |
    | LSTM    | 81.134        | 0.5868     | 0.7024          |
    | Bi-LSTM | 85.535        | 0.1236     | 0.6249          |
    | GRU     | 85.608        | 0.1750     | 0.6273          |

<br>

## Hyper-parameter Tuning

wandb의 sweep을 이용해 최적의 hyper-parameter를 찾으려고 했습니다.

### Hyper-parameter list

- batch-size : 16, 24, 32
- leraning_rate : 1e-05, 2e-05
- loss_function : Cross-Entropy, Focal Loss
- warm_up_ratio : 0, 0.1, 0.3, 0.6
- weight_decay : 0, 0.01
- lr_scheduler : Linear, Invsqrt, Cosine Annealing w/ Hard Restart

<br>

## Ensemble

- 리더보드 F1 및 AUPRC 점수를 기준으로 몇 개의 모델을 선별하여 앙상블을 적용하였습니다.
- Hard Voting, Soft Voting, Weighted Voting(Hard, Soft)을 시도했습니다.
- 그중 성능이 더 좋은 모델에 가중치를 주는 기법이었던 Weighted Voting(Hard)이 가장 성능이 좋았습니다.

| 최종 제출              | micro_f1 | auprc   |
| ---------------------- | -------- | ------- |
| LB Public Score        | 76.7790  | 81.5786 |
| LB Private Score (1등) | 76.3907  | 83.4108 |

<br>

## 팀 회고

### 좋았던 점

- Base Code를 pytorch_lightning, torchmetrics, huggingface를 이용해 기능을 모듈별로 분리해 프로젝트를 효율적으로 구조화할 수 있었다.
- 실험을 진행할 때 main branch에서 실험을 진행하지 않고 Branch를 나누어 실험해 실험이 효과적이었다는 것이 입증되면 main에 merge 하는 방식으로 Git을 활용하였다.
- 저번 기초 프로젝트에 비해서 데이터의 구조와 모델의 구조를 모두 깊게 뜯어보며 데이터와 모델 각각의 관점에서 다양한 실험을 진행할 수 있었다.
- offline meeting, online meeting을 적절히 섞어서 활용하며 프로젝트에 대한 이해도를 맞춰나갔다.

### 개선할 점

- 프로젝트 중반에 도달해서야 train/validation dataset, seed 등 실험을 위한 기초적인 세팅을 구조화하였으므로 프로젝트 초반에 진행한 실험을 다시 진행해야 했었다.
- commit message formatting, source code review, issue 등을 활용하여 더 체계적으로 저장소 관리를 하면 좋겠다.
- 프로젝트 마지막에 서로 맡은 부분에 대해 논의하고 결과를 공유하는 시간이 부족하여 효율적으로 실험을 진행하지 못했다.
- 초반에 학습 및 프로젝트 진행 속도를 맞추는 데 어려움을 겪었다.

<br>

## 참고 자료

[1] [Matching the Blanks: Matching the Blanks: Distributional Similarity for Relation Learning](https://aclanthology.org/P19-1279.pdf)

[2] [An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/pdf/2102.01373.pdf)

[3] [Unified Semantic Typing with Meaningful Label Inference](https://arxiv.org/pdf/2205.01826v1.pdf)

[4] [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/pdf/1901.05555.pdf)

[5] [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)

[6] [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/pdf/2004.10964.pdf)


