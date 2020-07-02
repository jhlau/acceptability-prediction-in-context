# Human Ratings
- Collected human ratings without context, with real context and with random context are in the csv files under **human-ratings**.
- These are filtered and preprocessed ratings with outliers removed (detailed in the last paragraph of section 2.1).
- Column names in the CSV files should be self-explanatory
  - _translated_ means whether the sentence has undergone round-trip machine translation, 1 indicates yes and 0 otherwise
  - _source-langage_ is the intermediate language of the round-trip machine translation (always "en" if _translated_=0)
  
# Publication

Jey Han Lau, Carlos Armendariz, Shalom Lappin, Matthew Purver, and Chang Shu (2020). [How Furiously Can Colorless Green Ideas Sleep? Sentence Acceptability in Context](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00315). In Transactions of the Association for Computational Linguistics, Vol 8, pages 296—310. 
