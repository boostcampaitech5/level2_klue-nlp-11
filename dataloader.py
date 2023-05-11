import pandas as pd
from tqdm.auto import tqdm
import pickle
import transformers
import torch
import pytorch_lightning as pl

ENTITY_MAP = {"ORG": "단체", "PER": "사람", "DAT": "날짜", "LOC": "위치", "POH": "기타", "NOH": "수량"}

class Dataset(torch.utils.data.Dataset):

    def __init__(self, inputs, targets=list(), ss_arr=None, os_arr=None):
        self.inputs = inputs
        self.targets = targets
        self.ss_arr = ss_arr
        self.os_arr = os_arr

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내옴
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행
        if len(self.targets) == 0:
            return (torch.tensor(self.inputs[idx]), torch.tensor([]), torch.tensor(self.ss_arr[idx]),
                    torch.tensor(self.os_arr[idx]))
        else:
            return (torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]), torch.tensor(self.ss_arr[idx]),
                    torch.tensor(self.os_arr[idx]))

    # 입력하는 개수만큼 데이터를 사용
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):

    def __init__(self, model_name, use_tokens, train_batch_size, val_batch_size, shuffle, train_path, dev_path,
                 test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.use_tokens = use_tokens

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        # special token : use ENTITY MARKERS – ENTITY START and entity type
        # source : https://aclanthology.org/P19-1279.pdf
        if self.use_tokens:
            special_tokens_dict = {
                'additional_special_tokens': [
                    '[SUBJ]', '[/SUBJ]', '[OBJ]', '[/OBJ]', "[PER]", "[ORG]", "[LOC]", "[POH]", "[DAT]", "[NOH]"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.target_columns = 'label'
        self.delete_columns = ['id']
        self.text_column = 'sentence'
        self.subj_column = 'subject_entity'
        self.obj_column = 'object_entity'

    def prepare_data(self) -> None:
        """ csv 파일을 경로에 맞게 불러 옵니다. """
        self.train_dataframe = pd.read_csv(self.train_path)
        self.val_dataframe = pd.read_csv(self.dev_path)
        self.test_dataframe = pd.read_csv(self.test_path)
        self.predict_dataframe = pd.read_csv(self.predict_path)

        # 학습데이터 준비
        self.train_inputs, self.train_targets, self.train_ss_arr, self.train_os_arr = self.preprocessing(
            self.train_dataframe)

        # 검증데이터 준비
        self.val_inputs, self.val_targets, self.val_ss_arr, self.val_os_arr = self.preprocessing(self.val_dataframe)
        # self.val_inputs, self.val_targets = self.train_inputs, self.train_targets

        # 평가데이터 준비
        self.test_inputs, self.test_targets, self.test_ss_arr, self.test_os_arr = self.preprocessing(
            self.test_dataframe)
        # self.test_inputs, self.test_targets = self.train_inputs, self.train_targets

        self.predict_inputs, self.predict_targets, self.predict_ss_arr, self.predict_os_arr = self.preprocessing(
            self.predict_dataframe)

    def preprocessing(self, data):
        # 안쓰는 컬럼 삭제
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열 리턴
        try:
            target_labels = data[self.target_columns].values

            def label_to_num(label):
                num_label = []
                with open('/opt/ml/level2_klue-nlp-11/utils/dict_label_to_num.pkl', 'rb') as f:
                    dict_label_to_num = pickle.load(f)
                for v in label:
                    num_label.append(dict_label_to_num[v])

                return num_label

            targets = label_to_num(target_labels)
        except Exception as e:
            print(e)
            targets = []
            
        # 텍스트 데이터 전처리
        inputs, ss_arr, os_arr = self.tokenize(data)

        return inputs, targets, ss_arr, os_arr

    def tokenize(self, dataframe):
        res, ss_arr, os_arr = [], [], []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 입력 문장의 entity 위치 파악
            _, subj_start, subj_end, subj_entity = [
                x.split(": ")[1].strip("'") for x in item[self.subj_column][1:-1].split(", '")
            ]
            _, obj_start, obj_end, obj_entity = [
                x.split(": ")[1].strip("'") for x in item[self.obj_column][1:-1].split(", '")
            ]
            
            subj_start = int(subj_start)
            subj_end = int(subj_end)
            obj_start = int(obj_start)
            obj_end = int(obj_end)
            
            tmp = []
            if self.use_tokens: #entity marker
                if subj_start < obj_start: #sub 객체의 위치가 obj 객체 위치보다 앞에 있을 때
                    tmp.extend([
                        item[self.text_column][:subj_start],
                        f'[SUBJ] [{subj_entity}] ' + item[self.text_column][subj_start:subj_end + 1] + ' [/SUBJ]',
                        item[self.text_column][subj_end + 1:obj_start], f'[OBJ] [{obj_entity}] ',
                        item[self.text_column][obj_start:obj_end + 1], ' [/OBJ]', item[self.text_column][obj_end + 1:]
                    ])
                elif subj_start > obj_start:
                    tmp.extend([
                        item[self.text_column][:obj_start],
                        f'[OBJ] [{obj_entity}] ' + item[self.text_column][obj_start:obj_end + 1] + ' [/OBJ]',
                        item[self.text_column][obj_end + 1:subj_start], f'[SUBJ] [{subj_entity}] ',
                        item[self.text_column][subj_start:subj_end + 1], ' [/SUBJ]',
                        item[self.text_column][subj_end + 1:]
                    ])
                else:
                    raise ValueError("subj-obj overlapped")
                
            else: #typed entity marker punctuation
                if subj_start < obj_start:
                    tmp.extend([
                        item[self.text_column][:subj_start],
                        f'@ ⊙ {ENTITY_MAP[subj_entity]} ⊙ ' + item[self.text_column][subj_start:subj_end + 1] + ' @',
                        item[self.text_column][subj_end + 1:obj_start], f'# ^ {ENTITY_MAP[obj_entity]} ^ ',
                        item[self.text_column][obj_start:obj_end + 1], ' #', item[self.text_column][obj_end + 1:]
                    ])
                elif subj_start > obj_start:
                    tmp.extend([
                        item[self.text_column][:obj_start],
                        f'# ^ {ENTITY_MAP[obj_entity]} ^ ' + item[self.text_column][obj_start:obj_end + 1] + ' #',
                        item[self.text_column][obj_end + 1:subj_start], f'@ ⊙ {ENTITY_MAP[subj_entity]} ⊙ ',
                        item[self.text_column][subj_start:subj_end + 1], ' @', item[self.text_column][subj_end + 1:]
                    ])
                else:
                    raise ValueError("subj-obj overlapped")
                
            ss = len(self.tokenizer(tmp[0], add_special_tokens=False)['input_ids']) + 1
            os = ss + len(self.tokenizer(tmp[1], add_special_tokens=False)['input_ids']) + len(
                self.tokenizer(tmp[2], add_special_tokens=False)['input_ids'])
            
            if subj_start > obj_start:
                ss, os = os, ss
            
            #query change
            # text_entity_verbalized = f'#{item[self.text_column][obj_start:obj_end + 1]}#는 ⊙{ENTITY_MAP[subj_entity]}⊙인 @{item[self.text_column][subj_start:subj_end + 1]}@의 ^{ENTITY_MAP[obj_entity]}^이다.'
            # text_entity_verbalized = f'#{item[self.text_column][obj_start:obj_end + 1]}#와 @{item[self.text_column][subj_start:subj_end + 1]}@의 관계는 ⊙{ENTITY_MAP[subj_entity]}⊙와 ^{ENTITY_MAP[obj_entity]}^의 관계다.'
            text_entity_verbalized = f'{item[self.text_column][obj_start:obj_end + 1]}와 {item[self.text_column][subj_start:subj_end + 1]}의 관계는 {ENTITY_MAP[subj_entity]}와 {ENTITY_MAP[obj_entity]}의 관계'
            
            offset = len(self.tokenizer(text_entity_verbalized, add_special_tokens=False)["input_ids"]) + 1
            ss += offset
            os += offset
            
            text = "".join(tmp)
            outputs = self.tokenizer(text_entity_verbalized,
                                     text,
                                     add_special_tokens=True,
                                     max_length=256,
                                     padding='max_length',
                                     truncation=True)
            res.append(outputs['input_ids'])
            ss_arr.append(ss)
            os_arr.append(os)
            
        return res, ss_arr, os_arr

    def setup(self, stage='fit'):
        if stage == 'fit':

            # 학습데이터 세팅
            self.train_dataset = Dataset(self.train_inputs, self.train_targets, self.train_ss_arr, self.train_os_arr)

            # 검증데이터 세팅
            self.val_dataset = Dataset(self.val_inputs, self.val_targets, self.val_ss_arr, self.val_os_arr)
        else:
            self.test_dataset = Dataset(self.test_inputs, self.test_targets, self.test_ss_arr, self.test_os_arr)

            self.predict_dataset = Dataset(self.predict_inputs, self.predict_targets, self.predict_ss_arr,
                                           self.predict_os_arr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.val_batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.val_batch_size)
    
