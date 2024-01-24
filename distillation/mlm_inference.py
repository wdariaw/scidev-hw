import argparse
import torch
import pytorch_lightning as pl
from data_modules import DataModule
from models import Teacher_Student_Model
from transformers import DataCollatorForLanguageModeling, BertTokenizer


def main(config):
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    data_module = DataModule(config, tokenizer, data_collator)
    model = Teacher_Student_Model(config, tokenizer)
    model.load_state_dict(torch.load(config.model_weights_path)["state_dict"])
    trainer = pl.Trainer(enable_progress_bar=False, accelerator="gpu", devices=1)
    trainer.predict(
        model,
        datamodule=data_module
    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pretrain distilled BERT.'
    )

    parser.add_argument('--student_hidden_size', type=int, default=312)
    parser.add_argument('--student_intermediate_size', type=int, default=1200)
    parser.add_argument('--student_num_hidden_layers', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--model_weights_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=3)

    parser = DataModule.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    main(args)