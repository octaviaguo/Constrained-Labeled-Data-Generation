# extract key words based on tf-idf
1. cd key_words_extract
2. mkdir data
3. python3 tfidf_extract.py --train_data=../data/10w-data/original.train --dev_data=../data/10w-data/original.dev --rate=0.5 \
python3 tfidf_extract.py --train_data=../data/10w-data/original.train --dev_data=../data/10w-data/original.dev --rate=0.25
4. use key_words_0.5.train (or key_words_0.25.train) and target_0.5.train (or target_0.25.train) as source and target of T5 training \
use key_words_bart_0.5.train (or key_words_bart_0.25.train) and target_0.5.train (or target_0.25.train) as source and target of BART training