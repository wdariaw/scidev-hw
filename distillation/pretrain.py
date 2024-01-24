import argparse
import torch
import pytorch_lightning as pl
from transformers import DataCollatorForLanguageModeling, BertTokenizer

from utils import seed_everything
from data_modules import DataModule
from models import Teacher_Student_Model


def main(config):
    seed_everything(42)
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    data_module = DataModule(config, tokenizer, data_collator)
    
    model = Teacher_Student_Model(config, tokenizer)
    trainer = pl.Trainer(max_epochs=config.epochs, val_check_interval=config.val_check_interval, accelerator="gpu", devices=1)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pretrain distilled BERT.'
    )

    parser.add_argument('--student_hidden_size', type=int, default=312)
    parser.add_argument('--student_intermediate_size', type=int, default=1200)
    parser.add_argument('--student_num_hidden_layers', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--val_check_interval', type=int, default=10420)
    
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    
    main(args)