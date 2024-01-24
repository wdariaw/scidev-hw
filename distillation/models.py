import torch
import torch.nn as nn
from torch.optim import AdamW

import pytorch_lightning as pl

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification
)

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


class Teacher_Student_Model(pl.LightningModule):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.student_config = BertConfig(
            hidden_size=args.student_hidden_size,
            intermediate_size=args.student_intermediate_size,
            num_hidden_layers=args.student_num_hidden_layers,
        )
        self.student = BertForMaskedLM(self.student_config)
        self.teacher = BertForMaskedLM.from_pretrained(args.model_name)
        self.config = args

        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        student_output = self.student(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            teacher_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
        return {'student': student_output, 'teacher': teacher_output}
    
    def predict_step(self, batch, batch_idx):
        input_ids = torch.Tensor(batch["input_ids"]).squeeze(0)
        attention_mask = torch.Tensor(batch["attention_mask"]).squeeze(0)

        output = self(input_ids=input_ids, attention_mask=attention_mask)
        
        model_names = ['teacher', 'student']
        unmasked_text = {model_name: []  for model_name in model_names}
        sentences = []
        
        for i, (sentence_input_ids, sentence_attention_mask) in enumerate(zip(input_ids, attention_mask)):
            pos = torch.argmax(torch.where(sentence_input_ids == self.tokenizer.mask_token_id, 1, 0), axis=-1)
            sentences.append(self.tokenizer.decode(sentence_input_ids.masked_select(sentence_attention_mask.type(torch.bool))[1:-1]))
            for model_name in model_names:
                most_likely_token_ids = torch.argmax(output[model_name].logits[i, pos, :])
                sentence_input_ids[pos] = most_likely_token_ids
                unmasked_text[model_name].append(self.tokenizer.decode(sentence_input_ids, skip_special_tokens=True))
                
        for i, sentence in enumerate(sentences):
            print(sentence)
            print(f'Teacher: {unmasked_text["teacher"][i]} \nStudent: {unmasked_text["student"][i]}\n')
         
        return unmasked_text["student"]
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output = self(input_ids=input_ids, attention_mask=attention_mask)
        
        soft_targets = nn.functional.softmax(output['teacher'].logits / self.config.T, dim=-1)
        soft_prob = nn.functional.log_softmax(output['student'].logits / self.config.T, dim=-1)
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (self.config.T**2)
        
        self.log("train_loss", soft_targets_loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": soft_targets_loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        output = self(input_ids=input_ids, attention_mask=attention_mask)
        
        soft_targets = nn.functional.softmax(output['teacher'].logits / self.config.T, dim=-1)
        soft_prob = nn.functional.log_softmax(output['student'].logits / self.config.T, dim=-1)
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (self.config.T**2)

        self.log("eval_loss", soft_targets_loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"eval_loss": soft_targets_loss}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_optimizer = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_parameters, self.config.lr)




class GLUEClassifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        
        if args.finetune_distilled:
            self.student_config = BertConfig(
                hidden_size=args.student_hidden_size,
                intermediate_size=args.student_intermediate_size,
                num_hidden_layers=args.student_num_hidden_layers
            )
            self.model = BertForSequenceClassification(self.student_config)
            checkpoint = torch.load(
                args.student_weights_path,
                map_location=torch.device('cpu')
            )

            for key in list(checkpoint['state_dict'].keys()):
                checkpoint['state_dict'][key.replace('student.', '')] = checkpoint['state_dict'].pop(key)

            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        self.lr = args.lr

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.log("train_loss", output.loss, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": output.loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.log("eval_loss", output.loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("eval_f1", f1_score(labels[:, 1].detach().cpu(), np.argmax(output.logits.detach().cpu(), axis=-1)), on_step=False, on_epoch=True)

        return {"eval_loss": output.loss}
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self(input_ids=input_ids, attention_mask=attention_mask)
        self.log('test_f1', f1_score(labels[:, 1].detach().cpu(), np.argmax(output.logits.detach().cpu(), axis=-1)), on_step=False, on_epoch=True)
        self.log('test_recall', recall_score(labels[:, 1].detach().cpu(), np.argmax(output.logits.detach().cpu(), axis=-1)), on_step=False, on_epoch=True)
        self.log('test_precision', precision_score(labels[:, 1].detach().cpu(), np.argmax(output.logits.detach().cpu(), axis=-1)), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_optimizer = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_parameters, self.lr)