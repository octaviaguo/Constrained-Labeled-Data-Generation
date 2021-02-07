from typing import List
from typing import Tuple
from torch.utils.data import Dataset
from transformers import T5Tokenizer, BartTokenizer
import nltk
import random
import pickle
import config

#EXTRA_TOKENS = ['à', 'è', 'ì', 'ò', 'ù', 'Á', 'É', 'Í', 'Ó', 'Ú', 'á', 'é', 'í', 'ó', 'ú', 'Â', 'Ê',
#                'â', 'ê', 'ô', 'Ã', 'Õ', 'ã', 'õ', 'ü']
EXTRA_TOKENS = ['<mask>']
WORDS = []#, 'Não', 'nós', 'só', 'Há']

def _add_noise(text, noise_vocab, noise=0.1):
    text = text.replace('<mask>', '↑')
    words = nltk.word_tokenize(text)
    for i in range(len(words)):
        #if words[i]=='<mask>':
        if words[i]=='↑':
            continue
        if random.random() < noise:
            words[i] = random.choice(noise_vocab)
            #if i + 1 < len(words) and random.random() < 0.5:
            #    words[i + 1] = random.choice(noise_vocab)
            #    if i + 2 < len(words) and random.random() < 0.5:
            #        words[i + 2] = random.choice(noise_vocab)
            #        if i + 3 < len(words) and random.random() < 0.5:
            #            words[i + 3] = random.choice(noise_vocab)

    result = ' '.join(words)
    result = result.replace('↑', '<mask>')
    return result

class MyDataset(Dataset):
    def __init__(self, text_pairs: List[Tuple[str]], tokenizer,
                 source_max_length: int = 32, target_max_length: int = 32):
        self.tokenizer = tokenizer
        self.text_pairs = text_pairs
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.noise_vocab = None
        self.noise = config.items['training']['noise']
        if self.noise > 0.0:
            self.noise_vocab = pickle.load(open(config.items['training']['noise_vocab'], 'rb'))
        model_type = config.items['model']['type']
        if 'bart' in model_type:
            self.end_token = self.tokenizer.eos_token
        else:
            self.end_token = ''  #<pad> is used as end_toekn in t5, so need not add it

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        source, target = self.text_pairs[idx]

        if self.noise > 0.0 :
            #print('origin:', source)
            source = _add_noise(source, self.noise_vocab, self.noise)
            #print('with noise:', source)                    

        source_modified = source + self.end_token
        target_modified = target + self.end_token

        source_tok = self.tokenizer.batch_encode_plus([source_modified], #add_special_tokens=True,
                                                      max_length=self.source_max_length,
                                                      padding = 'max_length',
                                                      return_tensors='pt', truncation=True)
        target_tok = self.tokenizer.batch_encode_plus([target_modified], #add_special_tokens=True,
                                                      max_length=self.target_max_length,
                                                      padding = 'max_length',
                                                      return_tensors='pt', truncation=True)

        #print(source_tok)
        return (source_tok['input_ids'][0], source_tok['attention_mask'][0], target_tok['input_ids'][0],
                target_tok['attention_mask'][0], source, target)


def create_adapted_tokenizer(model_name):
    if 't5' in model_name:
        #if config.items['data']['custom_tokenizer']:
        if False:
            print('read source data, use t5 custom tokenizer')
            tokenizer = T5Tokenizer.from_pretrained(config.items['data']['custom_tokenizer'])
        else:
            print('read source data, use t5 pretrained tokenizer')
            tokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        if config.items['data']['custom_tokenizer']:
            print('read source data, use bart custom tokenizer')
            tokenizer = BartTokenizer.from_pretrained(config.items['data']['custom_tokenizer'])
        else:
            print('read source data, use bart pretrained tokenizer')
            tokenizer = BartTokenizer.from_pretrained(model_name)

    #tokenizer = T5Tokenizer.from_pretrained(model_name)

    for word in WORDS:
        tokenizer.add_tokens(word)

    added_tokens = []
    for tok in EXTRA_TOKENS:
        enc = tokenizer.encode(tok)
        if 2 in enc:
            added_tokens.append(tok)
            tokenizer.add_tokens(tok)

    return tokenizer, added_tokens

'''
def create_ptt5_tokenizer():
    model_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    return tokenizer
'''
