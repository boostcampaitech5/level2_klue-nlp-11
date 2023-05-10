import pandas as pd
from transformers import RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from utils.seed import *  # seed setting module
from utils.callbacks import *
from dataloader import *
from models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

csv_file_path = "~/dataset/train/train.csv"
data = pd.read_csv(csv_file_path)

#텍스트 파일 
text_data = data["sentence"].tolist()
text_data_file = open("text_data.txt", "w", encoding="utf-8")
text_data_file.write("\n".join(text_data))
text_data_file.close()

# RobertaForMaskedLM 토크나이저 초기화
tokenizer = transformers.AutoTokenizer.from_pretrained("klue/roberta-large")

# 텍스트 데이터셋 준비
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text_data.txt",
    block_size=256  # 입력 시퀀스의 최대 길이
)

# 데이터 콜레이터 초기화
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # 마스킹된 언어 모델링 작업을 수행할 것
    mlm_probability=0.15  # 각 단어를 마스킹하는 확률
)

# RobertaForMaskedLM 모델 초기화
model = RobertaForMaskedLM.from_pretrained('klue/roberta-large')
model.parameters
model.to(device)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./tapt_model2",
    overwrite_output_dir=True,
    num_train_epochs=1,  # 사전 학습 횟수
    per_device_train_batch_size=32,  # 배치 크기 #
    save_steps=10,  # 일정 주기로 체크포인트 저장
    save_total_limit=3,  # 최대 체크포인트 파일 수
)

# 트레이너 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# TAPT 실행
trainer.train()

# 체크포인트 저장
best_model_checkpoint = get_last_checkpoint(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)  
model.save_pretrained(training_args.output_dir) 

#accumulation 실행해보기