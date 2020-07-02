# Requirements

- python3.6
- transformers==3.0.0
- pytorch-transformers=1.2.0 (for re-producing GPT-2 results)

Note: since pytorch-transformers were upgraded to transformers, the new GPT-2 results are little different. Most acceptability measures are about the same, but SLOR is no longer as bad (the odd results in the original paper). If you want to reproduce the GPT-2 results in the paper, use the XXX script.

The exact environment that was run on:
```
boto3==1.14.15
botocore==1.17.15
certifi==2020.6.20
chardet==3.0.4
click==7.1.2
cycler==0.10.0
dataclasses==0.7
docutils==0.15.2
filelock==3.0.12
future==0.18.2
idna==2.10
jmespath==0.10.0
joblib==0.16.0
kiwisolver==1.2.0
matplotlib==3.2.2
numpy==1.19.0
packaging==20.4
pandas==1.0.5
patsy==0.5.1
Pillow==7.2.0
pkg-resources==0.0.0
pyparsing==2.4.7
python-dateutil==2.8.1
pytorch-transformers==1.2.0
pytz==2020.1
regex==2020.6.8
requests==2.24.0
s3transfer==0.3.3
sacremoses==0.0.43
scipy==1.5.0
sentencepiece==0.1.91
six==1.15.0
statsmodels==0.11.1
tokenizers==0.8.0rc4
torch==1.2.0
torchvision==0.4.0
tqdm==4.47.0
transformers==3.0.0
urllib3==1.25.9
```

# Human Ratings
- Collected human ratings without context, with real context and with random context are in the csv files under **human-ratings**.
- These are filtered and preprocessed ratings with outliers removed (detailed in the last paragraph of section 2.1).
- Column names in the CSV files should be self-explanatory
  - _translated_ means whether the sentence has undergone round-trip machine translation, 1 indicates yes and 0 otherwise
  - _source-langage_ is the intermediate language of the round-trip machine translation (always "en" if _translated_=0)
  
# Publication

Jey Han Lau, Carlos Armendariz, Shalom Lappin, Matthew Purver, and Chang Shu (2020). [How Furiously Can Colorless Green Ideas Sleep? Sentence Acceptability in Context](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00315). In Transactions of the Association for Computational Linguistics, Vol 8, pages 296—310. 
