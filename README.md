# Relation Extraction

## Directory

level2_klue-nlp-11   
├── README.md   
├── config   
│   ├── config.py   
│   ├── config.yaml   
│   └── sweep_config.yaml   
├── dataloader.py   
├── inference.py   
├── models.py   
├── pretraining.py   
├── sweep.py   
├── train.py   
└── utils   
&nbsp;&nbsp;&nbsp;&nbsp;├── dict_label_to_num.pkl      
&nbsp;&nbsp;&nbsp;&nbsp;├── dict_num_to_label.pkl   
&nbsp;&nbsp;&nbsp;&nbsp;├── losses.py   
&nbsp;&nbsp;&nbsp;&nbsp;├── metrics.py   
&nbsp;&nbsp;&nbsp;&nbsp;├── seed.py   
&nbsp;&nbsp;&nbsp;&nbsp;└── utils.py   

## Usage

1. train.py: 
  * config 폴더의 config.yaml를 통해 원하는 hyperparameter 값들을 조정

2. sweep.py:
  * config 폴더의 sweep_config.yaml를 통해 원하는 hyperparameter 값들을 조정
