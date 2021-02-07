import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help='training data file')
parser.add_argument('-m', '--model_prefix', type=str, default='bart_tokenized_data', help='Prefix for model and vocab files')
args = parser.parse_args()

class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        self.tokenizer.train(trainer, paths)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)

data_file = args.input
tokenizer = BPE_token()# train the tokenizer model
tokenizer.bpe_train([data_file])# saving the tokenized data in our specified folder 
save_path = args.model_prefix
tokenizer.save_tokenizer(save_path)        
