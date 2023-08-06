# NanoBERTa
Predicting Nanobody Binding Epitopes Based on a Pretrained RoBERTa-based Model
# Model Description
The NanoBERTa is based on the Roberta architecture.
# Usage
  # Download data
    -To download the required data from the respective databases:
      ·Download unpaired heavy chain data specific to human sources from OAS for pretraining.
      ·Download PDB data(<3.0Å) from SAbDab for model fine-tuning.
  # Data preprocessing
    -To preprocessing the data，you can:
      ·Pretrain:
        Filter sequences that meet the criteria, example code:data-process/pretrain-data.py.
      ·Fine-tuning:
        Calculate the binding sites, example code:data-process/finetuning-data.py.
  # Training
      tokenizer：model/tokenizer
      ·Pretrain:
         example code:model/pre-train.py.
      ·Fine-tuning:
         example code:model/finetuning.py.
# Contact
For any questions or inquiries, please contact wangxin@sztu.edu.cn
