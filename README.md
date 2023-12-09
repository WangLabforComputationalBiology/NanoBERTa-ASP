# NanoBERTa-ASP
Predicting Nanobody Binding Epitopes Based on a Pretrained RoBERTa Model
# Model Description
The NanoBERTa-ASP is based on the RoBERTa architecture.
# Usage
  ## Download data
    -To download the required data from the respective databases:
      ·Download unpaired heavy chain data specific to human sources from OAS for pretraining.
      *https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/
      ·Download PDB data(<3.0Å) from SAbDab for model fine-tuning.
      *https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/
  ## Data preprocessing
    -To preprocessing the data，you can:
      ·Pretrain:
        Filter sequences that meet the criteria, example code:data-process/pretrain-data.py.
      ·Fine-tuning:
        Calculate the binding sites, example code:data-process/finetuning-data.py.
  ## Training
      tokenizer：model/tokenizer
      ·Pretrain:
         example code:model/pre-train.py.
      ·Fine-tuning:
         example code:model/finetuning.py.
         The Fine-tuning dataset is provided in folder NanoBERTa-ASP/assets in parquet format, you could open it by pandas package of Python.
# Contact
For any questions or inquiries, please contact Shangru Li (1372981079@qq.com) and wangxin@sztu.edu.cn
