"""Grid Beam Search"""
import torch
import sys
import os
import pickle
sys.path.append("../transformers/src/")
from transformers import BartTokenizer, BartForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel

from decode_gbs import GridBeamSearchDecoder
#from evaluate import get_bleu
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='bart', help='"bart" or "gpt2"')
parser.add_argument('--beam_size', type=int, default=4, help='beam_size')
parser.add_argument('--constrain_type', type=str, default='hard', help='"seq" or ""')
parser.add_argument('--process_size', type=int, default=32)
parser.add_argument('--input', type=str, default='', help='xxx xxx xxx')
parser.add_argument('--keys', type=str, default='', help='"xxx xxx,xxx",etc')
parser.add_argument('--pydev', type=bool, default=False, help='enable pydev debug.')
parser.add_argument('--print_log', type=bool, default=False, help='print debug log')


args = parser.parse_args()

def load_dict(dict_list, dict_pickle, tokenizer):
    if os.path.exists(dict_pickle):
        with open(dict_pickle, 'rb') as f:
            lan_dict = pickle.load(f)
    else:
        for f in dict_list:
            with open(f, 'r') as f:
                lan_dict = {}
                for i, line in enumerate(f):
                    item = line.split(' ')
                    p_tokens = tokenizer(' '+item[1].replace('\n','').strip(), return_tensors='pt', add_special_tokens=False)
                    if item[0] not in lan_dict:
                        lan_dict[item[0]] = []
                    lan_dict[item[0]].append(p_tokens['input_ids'][0].tolist())
                    print(i)
        with open(dict_pickle, 'wb') as f:
            pickle.dump(lan_dict, f)
    return lan_dict

if __name__ == '__main__':

    if args.pydev:
        import pydevd
        pydevd.settrace("localhost", port=5678) 

    if 'gpt2' in args.model_type:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'bos_token': '<s>'})
    else:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    if torch.cuda.is_available():
        model = model.cuda()

    if 'dict' in args.constrain_type:
        if 'gpt2' in args.model_type:
            lan_dict = load_dict(['../data/de_en_ner/de-en.txt'], '../data/de_en_ner/de-en_pretrained-gpt2.pickle', tokenizer)
        else:
            lan_dict = load_dict(['../data/de_en_ner/de-en.txt'], '../data/de_en_ner/de-en_pretrained-bart-base.pickle', tokenizer)

    elif 'hard' in args.constrain_type:
        lan_dict = None

    decoder = GridBeamSearchDecoder(model, tokenizer, lan_dict=lan_dict, beam_size=args.beam_size,
        model_type=args.model_type,
        constrain_type=args.constrain_type,
        process_size=args.process_size,
        print_log=args.print_log)
    if 'dict' in args.constrain_type:
        #cheap translation test
        if 'gpt2' in args.model_type:
            inp_en = '.'
        else:
            inp_en = '<mask> politics responded <mask> test <mask>'
        key_de = '<mask> politik <mask> sperre <mask>'
        hypotheses, all_preds2 = decoder.decode([inp_en], [key_de])
        for p in all_preds2:
            print('predicted: ', p)

    else:
        if 'gpt2' in args.model_type:
            inp1 = '.'
        else:
            inp1 = '<mask> politics responded <mask> test <mask>'
        key1 = ' test <mask> politics responded'
        
        inp2 = 'My friends are cool but they eat too many carbs.'
        key2 = 'my friends <mask> eat'

        hypotheses, all_preds1 = decoder.decode([inp2], None)
        print('1)predicted: ', all_preds1[0])
        
        #hypotheses, all_preds = decoder.decode([inp1,inp2], [key2,key1])
        #hypotheses, all_preds2 = decoder.decode([inp1,inp2], None)
        hypotheses, all_preds2 = decoder.decode([inp1], None)
        for p in all_preds2:
            print('2)predicted: ', p)
        
        #hypotheses, all_preds = decoder.decode([inp1, inp2], [key2, key2])

        hypotheses, all_preds3 = decoder.decode([inp1], [key1])
        print('3)predicted: ', all_preds3[0])

        hypotheses, all_preds3 = decoder.decode([inp2], [key2])
        print('4)predicted: ', all_preds3[0])

