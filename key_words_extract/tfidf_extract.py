import pickle
import fire
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=float, default=0.25)
parser.add_argument('--high_df_rate', type=float, default=0.25)
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--test_data', type=str, default=None)
parser.add_argument('--dev_data', type=str, default=None)
parser.add_argument('--pydev', type=bool, default=False)

args = parser.parse_args()

def get_texts(train_data, dev_data, test_data):
    conds, texts = {}, {}

    for split,src in zip(['train', 'dev', 'test'], [train_data, dev_data, test_data]):
        examples = []
        if src is None:
            continue
        with open(src, 'r') as f:
            for ind, line in enumerate(f):
                examples.append(line)

        conds[split], texts[split] = [], []
        for example in examples:
            texts[split].append(example)

    return texts


def get_vocab(texts, rate, extract_high_df):
    vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    features = vectorizer.fit_transform(texts).tocsc()

    vocab = vectorizer.get_feature_names()
    analyzer = vectorizer.build_analyzer()

    df = 1. / np.exp(vectorizer.idf_ - 1) * (len(texts) + 1) - 1

    word_value_list = []
    for i, word in enumerate(vocab):
        assert len(features[:, i].data) == int(round(df[i]))
        word_value_list.append(
            [word, np.mean(features[:, i].data), len(features[:, i].data)])
    word_value_list.sort(key=lambda t: t[1], reverse=not extract_high_df)

    total = sum([len(analyzer(text)) for text in texts])
    word_counter = {word: 0 for word in vocab}
    for text in texts:
        for word in analyzer(text):
            if word in word_counter:
                word_counter[word] += 1

    cnt = 0
    result_list = []
    for i, (word, _, df) in enumerate(word_value_list):
        result_list.append(word)
        cnt += word_counter[word]
        if cnt / total > rate:
            print(f'{i+1} words take {cnt / total} content.')
            break

    return result_list, analyzer


def main():
    if args.pydev:
        import pydevd
        pydevd.settrace("localhost", port=5678) 
    rate = args.rate

    texts = get_texts(args.train_data, args.dev_data, args.test_data)

    high_df_vocab, _ = get_vocab(texts['train'], rate=args.high_df_rate, extract_high_df=True)
    pickle.dump(high_df_vocab, open(f'data/high_df_vocab_{args.high_df_rate}.pickle', 'wb'))

    vocab, analyzer = get_vocab(texts['train'], rate=rate, extract_high_df=False)
    pickle.dump(vocab, open(f'data/vocab_{rate}.pickle', 'wb'))

    vocab_dict = {word: 1 for word in vocab}
    for split,src in zip(['train', 'dev', 'test'], [args.train_data, args.dev_data, args.test_data]):
        if src is None:
            continue
        print(f'extracting {split} set...')

        examples = []
        for text in texts[split]:
            key_words = []
            key_words_bart = []
            for word in analyzer(text):
                if word in vocab_dict:
                    key_words.append(word)
                    key_words_bart.append(word)
                elif len(key_words_bart)==0 or key_words_bart[-1]!='<mask>':
                    key_words_bart.append('<mask>')

            
            key_words_text=' '.join(key_words)
            key_words_bart_text = ' '.join(key_words_bart)

            examples.append({
                'key_words_text': key_words_text,
                'key_words_bart_text': key_words_bart_text,
                'original_text': text
            })


        target_file = open(
            f'data/target_{rate}.{split}', 'w')
        key_words_file = open(
            f'data/key_words_{rate}.{split}', 'w')
        key_words_bart_file = open(
            f'data/key_words_bart_{rate}.{split}', 'w')

        for example in examples:
            if len(example['key_words_text']) == 0:
                continue
            target_file.write(example['original_text'])
            key_words_file.write(example['key_words_text']+'\n')
            key_words_bart_file.write(example['key_words_bart_text']+'\n')

        key_words_file.flush()
        key_words_bart_file.flush()


if __name__ == '__main__':
    main()
