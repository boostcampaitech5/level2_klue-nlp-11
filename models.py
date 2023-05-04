import pickle as pickle
from typing import Any
from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                          RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, AutoModel,
                          RobertaModel)
import transformers
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils.losses import *
from utils.metrics import *
import torch.nn.functional as F


class FullyConnectedLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FullyConnectedLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.act(x)
        return self.linear(x)


class BaseModel(pl.LightningModule):

    def __init__(
            self,
            model_name,     # pretrained model name
            lr,
            weight_decay,
            loss_func,      # loss function type
            warmup_steps,   # warm up steps for learning rate scheduler
            total_steps,    # epochs * iteration per epoch, for linear decay scheduler
            LDAM_start=500, # DRW start step
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.config = AutoConfig.from_pretrained(model_name)
        self.LDAM_weight = torch.ones(30)

        # 사용할 모델을 호출
        self.plm = RobertaModel.from_pretrained(model_name, config=self.config)

        # Loss 계산을 위해 사용될 손실함수를 호출
        if loss_func == "CB":
            self.loss_func = CB_loss
        elif loss_func == "LDAM":
            self.loss_func = LDAMLoss(weight=self.LDAM_weight)
        elif loss_func == "CE":
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("CB, LDAM이외의 함수는 아직 지원x")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # warmup stage 있는 경우
        if self.warmup_steps is not None:
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                     num_warmup_steps=self.warmup_steps,
                                                                     num_training_steps=self.total_steps)
            return ([optimizer], [{
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False,
                'monitor': 'val_loss',
            }])
        # warmup stage 없는 경우
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
            return [optimizer], [scheduler]


# https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction 참조
class ClinicalTransformer(BaseModel):

    def __init__(
            self,
            model_name,     # pretrained model name
            lr,
            weight_decay,
            loss_func,      # loss function type
            warmup_steps,   # warm up steps for learning rate scheduler
            total_steps,    # epochs * iteration per epoch, for linear decay scheduler
            LDAM_start=500, # DRW start step
    ):
        super().__init__(model_name, lr, weight_decay, loss_func, warmup_steps, total_steps, LDAM_start)
        self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = range(32000, 32004)
        self.scheme = 2

        # 추가된 special tokens에 따른 embedding layer size 조정
        self.plm.resize_token_embeddings(32010)

        self.dense1 = FullyConnectedLayer(self.config.hidden_size * 5, 768, 0.1)
        self.dense2 = FullyConnectedLayer(768, 30, 0.1)
        self.classifier = FullyConnectedLayer(30, 30, 0.1)

    @staticmethod
    def special_tag_representation(seq_output, input_ids, special_tag):
        spec_idx = (input_ids == special_tag).nonzero(as_tuple=False)

        temp = []
        for idx in spec_idx:
            temp.append(seq_output[idx[0], idx[1], :])
        tags_rep = torch.stack(temp, dim=0)

        return tags_rep

    def output2logits(self, pooled_output, seq_output, input_ids):
        if self.scheme == 1:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag3]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
        elif self.scheme == 2:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
        elif self.scheme == 3:
            seq_tags = []
            for each_tag in [self.spec_tag1, self.spec_tag3]:
                seq_tags.append(self.special_tag_representation(seq_output, input_ids, each_tag))
            new_pooled_output = torch.cat(seq_tags, dim=1)
        else:
            new_pooled_output = pooled_output

        # logits = self.base_classifier(self.drop_out(new_pooled_output))

        return new_pooled_output

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs):

        outputs = self.plm(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states)

        seq_output = outputs[0]
        pooled_output = outputs[1]
        logits = self.output2logits(pooled_output, seq_output, input_ids)
        logits = self.dense1(logits)
        logits = self.dense2(logits)
        logits = self.classifier(logits)

        return logits

    def training_step(self, batch, batch_idx):
        # re-balancing ldam weight
        if isinstance(self.loss_func, LDAMLoss) and self.trainer.global_step == self.LDAM_start:
            beta = 0.9999
            effectice_num = 1.0 - torch.power(beta, num_per_cls)
            w = (1.0 - beta) / effectice_num
            w = w / w.sum() * 30
            self.LDAM_weight = w
            self.loss_func = LDAMLoss(weight=self.LDAM_weight)
        x, y = batch
        logits = self(x)
        loss = self.loss_func(y, logits)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(y, logits)
        preds = logits.argmax(-1)
        self.log("val_loss", loss)

        self.log("val_f1", klue_re_micro_f1(preds, y) * 100)
        self.log("val_auprc", klue_re_auprc(logits, y) * 100)
        self.log("val_acc", klue_re_acc(preds, y) * 100)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(-1)
        self.log("test_f1", klue_re_micro_f1(preds, y) * 100)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits


class TypedEntityMarkerPuncModel(BaseModel):

    def __init__(
        self,
        model_name,     # pretrained model name
        lr,
        weight_decay,
        loss_func,      # loss function type
        warmup_steps,   # warm up steps for learning rate scheduler
        total_steps,    # epochs * iteration per epoch, for linear decay scheduler
        LDAM_start=500,
    ):
        super().__init__(model_name, lr, weight_decay, loss_func, warmup_steps, total_steps)
        self.save_hyperparameters()

        self.LDAM_start = LDAM_start
        self.dense1 = FullyConnectedLayer(self.config.hidden_size * 5, 768, 0.1)
        self.dense2 = FullyConnectedLayer(768, 30, 0.1)

        self.classifier = nn.Sequential(nn.Linear(2 * self.config.hidden_size, self.config.hidden_size), nn.ReLU(),
                                        nn.Dropout(p=0.1), nn.Linear(self.config.hidden_size, 30))

    def forward(self, input_ids=None, ss=None, os=None, **kwargs):

        outputs = self.plm(input_ids)

        seq_output = outputs[0]
        # pooled_output = outputs[1]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = seq_output[idx, ss]
        os_emb = seq_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)

        return logits

    def training_step(self, batch, batch_idx):
        # re-balancing ldam weight
        if isinstance(self.loss_func, LDAMLoss) and self.trainer.global_step == 1000:
            beta = 0.9999
            effectice_num = 1.0 - torch.pow(beta, torch.tensor(num_per_cls))
            w = (1.0 - beta) / effectice_num
            w = w / w.sum() * 30
            self.LDAM_weight = w
            self.loss_func = LDAMLoss(weight=self.LDAM_weight)
        x, y, ss, os = batch
        logits = self(x, ss=ss, os=os)
        loss = self.loss_func(y, logits)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ss, os = batch
        logits = self(x, ss=ss, os=os)
        loss = self.loss_func(y, logits)
        preds = logits.argmax(-1)
        self.log("val_loss", loss)

        self.log("val_f1", klue_re_micro_f1(preds, y) * 100)
        # self.log("val_auprc", klue_re_auprc(logits, y) * 100)
        # self.log("val_acc", klue_re_acc(preds, y) * 100)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, ss, os = batch
        logits = self(x, ss=ss, os=os)
        preds = logits.argmax(-1)
        self.log("test_f1", klue_re_micro_f1(preds, y) * 100)

    def predict_step(self, batch, batch_idx):
        x, y, ss, os = batch
        logits = self(x, ss=ss, os=os)
        return logits
