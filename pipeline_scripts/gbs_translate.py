"""Grid Beam Search"""
import torch
import os
import codecs
from collections import defaultdict
from collections import deque
import sys
sys.path.append("../transformers/src/")
from transformers import T5Tokenizer, BartTokenizer, GPT2Tokenizer
from transformers.modeling_t5 import T5ForConditionalGeneration, T5Config
from transformers.modeling_bart import BartForConditionalGeneration, BartConfig
from transformers.modeling_gpt2 import GPT2LMHeadModel
sys.path.append("../grid_beam_search")
from decode_gbs import GridBeamSearchDecoder
sys.path.append("./my-python-translate")
from translate import Translator
from utils import readconll
from distutils.util import strtobool

import json
import pickle
#from evaluate import get_bleu
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument('--n_best', type=int, default=1)
parser.add_argument('--gpu_index', type=str, default='0')
parser.add_argument('--process_size', type=int, default=32)
parser.add_argument('--high_mem', type=int, default=0)
parser.add_argument('--low_mem', type=int, default=0)
parser.add_argument("--use_cache", 
    type=lambda x:bool(strtobool(x)),
    nargs='?', const=True, default=True)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--dump_index', type=int, default=-1)
parser.add_argument('--dump_file', type=str, default='')
parser.add_argument('--debug', type=bool, default=False)

parser.add_argument('--src_path', type=str, default='')
parser.add_argument('--src_format', type=str, default='')
parser.add_argument('--output_file', type=str, default='')
parser.add_argument('--out_sub_name', type=str, default='')
parser.add_argument('--log_file', type=str, default='')

parser.add_argument('--pydev', type=bool, default=False, help='enable pydev debug.')
parser.add_argument('--print_log', type=str, default=None)

args = parser.parse_args()
print('use_cache is ', args.use_cache)
config_file_path = args.config
config = json.load(open(config_file_path, 'r'))
base_name = args.config.split('/')[-1].replace('.json','')

train_type = config['model']['train_type']
model_type = config['model']['type']
model_path = config['model']['model']

#always True if used for NER
key_filter_on_O = config['data'].get('key_filter_on_O', True)
key_filter_on_non_group = config['data'].get('key_filter_on_non_group', True)
#
input_filter_vocab = config['data']['input_filter_vocab']
key_filter_vocab = config['data']['key_filter_vocab']
key_filter_label = config['data']['key_filter_label']
dict_list = config['data']['dict_list']
dict_cache = config['data']['dict_cache']
case_sensitive = config['data'].get('case_sensitive', True)
use_srilm = config['data']['use_srilm']
target_lan = config['data']['target_lan']
pre_proc_type = config['data'].get('pre_process_type','')
tmp_path = config['data'].get('tmp_path', './tmp/')
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

use_direct_tran = config['decode'].get('use_direct_tran', False)
num_beams = config['decode']['num_beams']
constrain_type = config['decode']['constrain_type']
max_unconstraint_len = config['decode']['max_unconstraint_len']
unkey_uc_ratio = config['decode'].get('unkey_uc_ratio',0.0)
do_sample = config['decode']['do_sample']
repetition_penalty = config['decode']['repetition_penalty']
no_repeat_ngram_size = config['decode']['no_repeat_ngram_size']
policy = config['decode'].get('policy', {})
not_translate_labels = policy.get('not_translate_labels',[])
ne_group_size = policy.get('ne_group_size',0)
reverse_order = config['decode'].get('reverse_order', False)
print('unkey_uc_ratio %d max_unconstraint_len %d'%(unkey_uc_ratio,max_unconstraint_len))
print('reverse_order:', reverse_order)
if len(args.output_file)==0:
    base_path = 'data/%s/'%target_lan
    output_file0 = base_path+base_name+'.train' + args.out_sub_name
    if reverse_order:
        rev_output_file0 = base_path+base_name+'.train.revout'+ args.out_sub_name
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    log_file = output_file0+('.dumplog' if args.dump_index>=0 else '.log')
    log_f=open(log_file, 'a' if args.start_index>0 else 'w')
    if args.dump_index>=0:
        dump_file = output_file0+'.dump'
        dump_f=open(dump_file, 'w')
else:
    output_file0=args.output_file
    if len(args.log_file)>0:
        log_f=open(args.log_file, 'a' if args.start_index>0 else 'w')
    else:
        log_f=None
    if len(args.dump_file)>0:
        dump_f=open(args.dump_file, 'w')
    else:
        dump_f=open

tot_score=0.0
score_count=0

print(args.dump_index)
class TestInputData(object):
    def __init__(self, batch_sent, batch_key_phrases):
        self.batch_inp_sent = batch_sent
        self.batch_keys = []
        if batch_key_phrases is not None:
            for key_phrases in batch_key_phrases:
                keys = []
                key_phrases = key_phrases.split('<mask>')
                for p in key_phrases:
                    k = [w for w in p.split(' ') if len(w)>0]
                    if len(k)>0:
                        keys.append(k)
                self.batch_keys.append(keys)        
        else:
            self.batch_keys = [None]*len(self.batch_inp_sent)

    def __len__(self):
        return len(self.batch_inp_sent)

    def __getitem__(self, idx):
        print('input: ', self.batch_inp_sent[idx])
        print('keys: ',self.batch_keys[idx])
        return (self.batch_inp_sent[idx], self.batch_keys[idx])

class CheapTranInputData(object):
    def __init__(self, src_path, flist, min_size=6, input_filter_vocab=None, tokenizer=None):
        is_f2e = config['data']['target_lan']=='eng'
        trg_filter_vocab = config['data'].get('trg_filter_vocab',None)
        if trg_filter_vocab is not None and os.path.exists(trg_filter_vocab):
            with open(trg_filter_vocab, 'rb') as f:
                trg_filter_vocab = pickle.load(f)

        self.labels = [
        '[LOC]',
        '[PER]',
        '[ORG]',
        '[MISC]',
        '[GPE]',
        '[O]',
        ]
        self.un_labeled = '[O]'

        self.batch_src_sent = []
        self.batch_inp_sent = []
        self.batch_keys = []
        self.batch_labels = []
        self.batch_un_key_count = []
        self.batch_ct_words = []
        self.batch_ct_labels = []
        self.filter_token_vocab = None

        dct = None
        self.token_dict = None
        if os.path.exists(dict_cache):
            with open(dict_cache, 'rb') as f:
                dict_loaded = pickle.load(f)
                self.token_dict = dict_loaded['token_dict']
                e2f = dict_loaded['e2f']
                f2e = dict_loaded['f2e']
                pairs = dict_loaded['pairs']     
                dct = defaultdict(lambda: defaultdict(float))
                if is_f2e:
                    for k in list(f2e.keys()):

                        scores = [(w, pairs[(w,k)]) for w in f2e[k]]

                        t1 = float(sum([p[1] for p in scores]))
                        t1 = max(0.1, t1)
                        nscores = sorted([(p[0], p[1] / t1) for p in scores], key=lambda p: p[1])

                        for p in nscores:
                            dct[k][p[0]] += p[1]
                else:
                    for k in list(e2f.keys()):

                        scores = [(w, pairs[(k,w)]) for w in e2f[k]]

                        t1 = float(sum([p[1] for p in scores]))
                        t1 = max(0.1, t1)
                        nscores = sorted([(p[0], p[1] / t1) for p in scores], key=lambda p: p[1])

                        for p in nscores:
                            dct[k][p[0]] += p[1]

        self.tt = Translator(None, None, None, lexname=flist[0], dict_cache=dct , is_f2e=is_f2e)

        if self.token_dict is None:
            self.generate_token_dict(tokenizer)

        if False:
            self.tt.translate_file(src_path, 'tmp/output.txt', 'conll')
        else:
            use_target_input_filter = config['data']['target_lan'] == config['data']['input_filter_lan']
            key_filter_pos = config['data']['key_filter_pos']
            vocab_filter_pos = config['data']['vocab_filter_pos']

            if os.path.isdir(src_path):
                f_list = os.listdir(src_path)
                f_list = sorted(f_list)
                fnames = [src_path + "/" + f for f in f_list]
            else:
                fnames = [src_path]
            
            if 'bart' in model_type:
                end_token = '</s>'
            else:
                end_token = ''  #<pad> is used as end_toekn in t5, so need not add it
            lines = []
            for fname in fnames:
                if '2col' in fname:
                    with codecs.open(fname,"r","utf-8") as f:
                        for i, line in enumerate(f):
                            item = line.split(' ')
                            #print(item)
                            if len(item)>=2:
                                lines.append('%s\tx\tx\tx\tx\t%s\tx\tx\tx\t'%(item[1].replace('\n',''),item[0]))
                            else:
                                lines.append('')
                else:
                    lines += readconll(fname)
                    lines += ['']
            
            if vocab_filter_pos is not None or trg_filter_vocab is not None:
                filter_vocab_cache = config['data'].get('filter_vocab_cache','')
                if len(filter_vocab_cache)==0:
                    filter_vocab_cache = tmp_path+'/'+base_name+'.filter_vocab_cache'
                if os.path.exists(filter_vocab_cache):
                    with open(filter_vocab_cache, 'rb') as f:
                        self.filter_token_vocab = pickle.load(f)
                        vocab_filter_pos = None
                        trg_filter_vocab = None

            (self.batch_src_sent, self.batch_cheaptran_sent, self.batch_inp_sent, self.batch_keys, self.batch_labels, self.batch_un_key_count,
                self.batch_ct_words, self.batch_ct_labels, self.batch_group,
                tran_trg_filter_vocab) = self.tt.translate(lines,
                key_filter_label=key_filter_label, key_filter_pos=key_filter_pos, vocab_filter_pos=vocab_filter_pos,
                input_filter_vocab=input_filter_vocab, use_target_input_filter=use_target_input_filter,
                end_token=end_token, custom_dict=self.token_dict, ne_group_size=ne_group_size,
                max_index=args.dump_index, pre_proc=pre_proc_type)
            print('tran_trg_filter_vocab len: ', len(tran_trg_filter_vocab))
            if len(tran_trg_filter_vocab)>0:
                tran_trg_filter_vocab.append(',')
                tran_trg_filter_vocab.append('.')
                trg_filter_vocab = tran_trg_filter_vocab
            if trg_filter_vocab is not None:
                self.filter_token_vocab = []
                for w in trg_filter_vocab:
                    print(w)
                    p_tokens = tokenizer(' '+w, return_tensors='pt', add_special_tokens=False)
                    self.filter_token_vocab.append(p_tokens['input_ids'][0].tolist())
                print('tokenizer.eos_token_id:', tokenizer.eos_token_id)
                self.filter_token_vocab.append([tokenizer.eos_token_id])

                with open(filter_vocab_cache, 'wb') as f:
                    pickle.dump(self.filter_token_vocab, f)
            if self.filter_token_vocab is not None:
                print('filter_token_vocab len: ', len(self.filter_token_vocab))


    def __len__(self):
        if args.dump_index >= 0:
            return 1

        return len(self.batch_inp_sent)

    def __getitem__(self, idx):
        #return (self.batch_inp_sent[idx], self.batch_keys[idx])
        if args.dump_index >= 0:
            idx = args.dump_index
        print('%d/%d======================'%(idx, len(self.batch_inp_sent)))
        if log_f is not None:
            log_f.write('%d/%d======================'%(idx, len(self.batch_inp_sent))+'\n')
            log_f.write('src: ' + self.batch_src_sent[idx]+'\n')
            log_f.write('cheap tran: ' + self.batch_cheaptran_sent[idx]+'\n')
            if 'gpt2' not in model_type:
                log_f.write('input: '+ self.batch_inp_sent[idx]+'\n')
            log_f.write('keys: '+ ','.join(['['+','.join([key for key in keys])+
                ' (%s)<%d>]'%(l[0][1:-1],g) for keys,l,g in zip(self.batch_keys[idx],self.batch_labels[idx],self.batch_group[idx])])+
                ' un_key_count:'+str(self.batch_un_key_count[idx])+'\n')
        else:
            print('src: ', self.batch_src_sent[idx])
            print('cheap tran: ', self.batch_cheaptran_sent[idx])
            if 'gpt2' not in model_type:
                print('input: ', self.batch_inp_sent[idx])
            print('keys: ',self.batch_keys[idx])
            print('label: ',self.batch_labels[idx])
        if 'gpt2' in model_type:
            return ('<skip>' if (use_direct_tran or self.batch_inp_sent[idx]=='<skip>') else '.', self.batch_keys[idx], self.batch_labels[idx], self.batch_un_key_count[idx],
                self.batch_ct_words[idx], self.batch_ct_labels[idx], self.batch_group[idx])
        else:
            return ('<skip>' if use_direct_tran else self.batch_inp_sent[idx], self.batch_keys[idx], self.batch_labels[idx], self.batch_un_key_count[idx],
                self.batch_ct_words[idx], self.batch_ct_labels[idx], self.batch_group[idx])

    def generate_token_dict(self, tokenizer):
        self.token_dict = {}
        tot = len(self.tt.dct)
        for i, srcphrase in enumerate(self.tt.dct):
            opts = dict(self.tt.dct[srcphrase])
            sbest = sorted(list(opts.items()), key=lambda p: -p[1]) #best first
            for itm in sbest:
                p_tokens = tokenizer(' '+itm[0], return_tensors='pt', add_special_tokens=False)
                if srcphrase not in self.token_dict:
                    self.token_dict[srcphrase] = []
                self.token_dict[srcphrase].append([itm[0],itm[1],p_tokens['input_ids'][0].tolist()])
            print('generate token_dict %d/%d'%(i,tot))

        self.dict_dump = {'token_dict': self.token_dict, 'e2f':self.tt.e2f, 'f2e':self.tt.f2e, 'pairs': self.tt.pairs}
        with open(dict_cache, 'wb') as f:
            pickle.dump(self.dict_dump, f)

    def get_tran_words(self, ph):
        return self.token_dict.get(ph, [[ph,0.0]])


class SimpleTranInputData(object):
    def __init__(self, src_path, dict_list, min_size=6, input_filter_vocab=None, tokenizer=None):
        self.filter_token_vocab = None
        self.load_dict(dict_list, dict_cache, tokenizer)

        self.labels = [
        '[LOC]',
        '[PER]',
        '[ORG]',
        '[MISC]',
        '[GPE]',
        '[O]',
        ]
        self.un_labeled = '[O]'

        use_target_input_filter = config['data']['target_lan'] == config['data']['input_filter_lan']
        key_filter_pos = config['data']['key_filter_pos']
        vocab_filter_pos = config['data']['vocab_filter_pos']
        trg_filter_vocab = config['data'].get('trg_filter_vocab',None)
        if trg_filter_vocab is not None and os.path.exists(trg_filter_vocab):
            with open(trg_filter_vocab, 'rb') as f:
                trg_filter_vocab = pickle.load(f)
        if vocab_filter_pos is not None or trg_filter_vocab is not None:
            filter_vocab_cache = config['data'].get('filter_vocab_cache','')
            if len(filter_vocab_cache)==0:
                filter_vocab_cache = tmp_path+'/'+base_name+'.filter_vocab_cache'
            if os.path.exists(filter_vocab_cache):
                with open(filter_vocab_cache, 'rb') as f:
                    self.filter_token_vocab = pickle.load(f)
                    vocab_filter_pos = None
                    trg_filter_vocab = None
        tran_trg_filter_vocab = self.translate(src_path, min_size, input_filter_vocab,
            use_target_input_filter, vocab_filter_pos, key_filter_pos)
        if len(tran_trg_filter_vocab)>0:
            tran_trg_filter_vocab.append(',')
            tran_trg_filter_vocab.append('.')
            trg_filter_vocab = tran_trg_filter_vocab
        if trg_filter_vocab is not None:
            self.filter_token_vocab = []
            for w in trg_filter_vocab:
                print(w)
                p_tokens = tokenizer(' '+w, return_tensors='pt', add_special_tokens=False)
                self.filter_token_vocab.append(p_tokens['input_ids'][0].tolist())
            print('tokenizer.eos_token_id:', tokenizer.eos_token_id)
            self.filter_token_vocab.append([tokenizer.eos_token_id])

            with open(filter_vocab_cache, 'wb') as f:
                pickle.dump(self.filter_token_vocab, f)
        if self.filter_token_vocab is not None:
            print('filter_token_vocab len: ', len(self.filter_token_vocab))

    def translate(self, src_path, min_size, input_filter_vocab, use_target_input_filter,
        vocab_filter_pos, key_filter_pos):

        self.batch_src_sent = []
        self.batch_inp_sent = []
        self.batch_keys = []
        self.batch_labels = []
        self.batch_un_key_count = []
        self.batch_ct_words = []
        self.batch_ct_labels = []  
        self.batch_group = []      
        wait_entity = True
        one_sentence = []
        keys_one_sent = []
        label_one_sent = []
        one_src_sentence = []
        pos_one_sent = []
        un_key_count = 0
        ct_words = []
        ct_labels = []
        tran_trg_filter_vocab = []
        src_count = 0
        if 'bart' in model_type:
            end_token = '</s>'
        else:
            end_token = ''  #<pad> is used as end_toekn in t5, so need not add it

        if key_filter_pos is not None:
            if key_filter_pos[0]=='<exclude>':
                key_exclude_pos = key_filter_pos[1:]
                key_include_pos = ['all']
            else:
                key_include_pos = key_filter_pos
                key_exclude_pos = []
        else:
            key_include_pos = ['all']
            key_exclude_pos = []

        """
        Translate by word a bunch of CoNLL files found in folder, and write the out to outfolder. Use the files in flist
        as the word mappings.
        
        folder -- a folder containing CoNLL files to be translated
        outfolder -- where the translated files will go
        flist -- a list of files containing word mappings, each of the format: source \t target. Order this list by trustworthiness from most to least.
        """
        if os.path.isdir(src_path):
            f_list = os.listdir(src_path)
            f_list = sorted(f_list)
            fnames = [src_path + "/" + f for f in f_list]
        else:
            fnames = [src_path]

        fixed = 0
        total = 0
        unfixed = defaultdict(int)
        for fname in fnames:
            print(fname)
            with codecs.open(fname, "r", "utf-8") as f:
                lines = f.readlines()
            queue = deque()
            for line in lines:            
                sline = line.split("\t")
                #print(sline)
                if len(sline)>=4:
                    if args.src_format=='col4':
                        src_word_raw = sline[0].strip()
                        label = sline[3].strip()
                        pos = sline[1].strip()
                    else:
                        src_word_raw = sline[5].strip()
                        label = sline[0].strip()
                        pos = sline[4].strip()

                    total += 1
                    one_src_sentence.append(src_word_raw)
                    src_word = src_word_raw.lower()
                    src_count += 1
                    #if src_word in self.wordmap and label[2:] not in not_translate_labels:
                    if src_word in self.token_dict and label[2:] not in not_translate_labels:
                        fixed += 1
                        #choices = self.wordmap[src_word]
                        choices = [wi[0] for wi in self.token_dict[src_word]]

                        # select the current word according to a language model
                        # conditioned on prior 2 words.
                        # TODO: add as an option
                        if use_srilm and len(choices) > 1 and len(queue) == 2:
                            ngramlist = []
                            for c in choices:
                                ngramlist.append(" ".join(queue) + " " + c + "\n")
                            
                            result = util.call_lm(ngramlist)
                            bestline = result.split("\n")[0]
                            trg_word = bestline.split()[-1].decode("utf8")
                        else:
                            # choose the first one
                            trg_word = choices[0]

                        if label=='O' and vocab_filter_pos is not None and pos in vocab_filter_pos:
                            for witm in choices:
                                if witm[0] not in tran_trg_filter_vocab:
                                    tran_trg_filter_vocab.append(witm[0])
                            #if trg_word not in tran_trg_filter_vocab:
                            #    tran_trg_filter_vocab.append(trg_word)

                    else:
                        #if label[2:] in not_translate_labels:
                        #    print(label[2:]+ '|', not_translate_labels, 'nnnnnn', src_word)
                        trg_word = src_word
                        unfixed[src_word] += 1

                    if case_sensitive:
                        if src_word_raw.isupper():
                            trg_word = trg_word.upper()
                        elif src_word_raw[0].isupper():
                            trg_word = trg_word[0].upper() + trg_word[1:]
                        #if src_word!=src_word_raw:
                        #    print('src_word_raw:', src_word_raw, src_word, trg_word)

                    ct_words.append(trg_word)
                    ct_labels.append(label)

                    queue.append(trg_word)
                    if len(queue) > 2:
                        queue.popleft()                   

                    if 'dict' in constrain_type:
                        key_word = src_word_raw if case_sensitive else src_word
                    else:
                        key_word = trg_word
                    #if one_sentence[-1]=='.':
                    #    sentence_done = True

                    if use_target_input_filter:
                        if input_filter_vocab is None or trg_word in input_filter_vocab:
                            one_sentence.append(trg_word)
                        else:
                            if len(one_sentence)==0 or one_sentence[-1]!='<mask>':
                                one_sentence.append('<mask>')
                    else:
                        if input_filter_vocab is None or src_word in input_filter_vocab:
                            one_sentence.append(trg_word)
                        else:
                            if len(one_sentence)==0 or one_sentence[-1]!='<mask>':
                                one_sentence.append('<mask>')

                    ### Keys  ==> IOB1
                    if label != 'O':
                        if label[0]=='B': # start a new entity
                            #if len(keys_one_sent):
                            keys_one_sent.append([key_word])
                            label_one_sent.append([label[2:]])
                            pos_one_sent.append([pos])
                            wait_entity = False
                        elif label[0]=='I': # start/in a new entity
                            if wait_entity:
                                keys_one_sent.append([key_word])
                                label_one_sent.append([label[2:]])
                                pos_one_sent.append([pos])
                            else:
                                keys_one_sent[-1].append(key_word)
                                label_one_sent[-1].append(label[2:])
                                pos_one_sent[-1].append(pos)
                            wait_entity = False
                    else:
                        keys_one_sent.append([key_word])
                        label_one_sent.append([label])
                        pos_one_sent.append([pos])
                        wait_entity = True                    
                    
                else:
                    queue = deque()
                    wait_entity = True

                    if ne_group_size>0:
                        siz = len(keys_one_sent)
                        group = [0]*siz
                        ne_idx = [ii for ii,lab in enumerate(label_one_sent) if lab[0]!='O']
                        #print(keys_one_sent, label_one_sent, ne_idx)
                        for ii in ne_idx:
                            ss = max([0,ii-ne_group_size])
                            ee = min([ii+ne_group_size+1,siz])
                            if sum(group[ss:ee])>0:
                                #aaa=True
                                #print(label_one_sent, ne_idx)
                                idx = max(group[ss:ee])
                            else:
                                idx = ii+1 #make sure idx is not 0
                            for kk in range(ss,ee):
                                group[kk]=idx
                    else:
                        group = [-1]*len(keys_one_sent)


                    keys_one_sent_f, label_one_sent_f, group_one_sent_f = [],[],[]
                    for ks,ls,g,ps in zip(keys_one_sent, label_one_sent, group, pos_one_sent):
                        if (key_filter_on_O and ls[0]!='O') or (key_filter_on_non_group and g>0):
                            keys_one_sent_f.append(ks)
                            label_one_sent_f.append(['['+l+']' for l in ls])
                            group_one_sent_f.append(g)
                        elif (key_filter_label is None or ls[0] in key_filter_label) and \
                            ('all' in key_include_pos or ps[0] in key_include_pos) and \
                            (ps[0] not in key_exclude_pos):
                                #if key_word!=',' and key_word!='.':
                            keys_one_sent_f.append(ks)
                            label_one_sent_f.append(['['+l+']' for l in ls])
                            group_one_sent_f.append(g)
                        else:
                            un_key_count += 1


                    if src_count>=min_size:
                        self.batch_inp_sent.append(' '.join(one_sentence)+'<mask>'+end_token)
                        #self.batch_inp_sent.append('<mask>')
                    else:
                        self.batch_inp_sent.append('<skip>')
                   
                    self.batch_keys.append(keys_one_sent_f)
                    self.batch_labels.append(label_one_sent_f)
                    self.batch_src_sent.append(' '.join(one_src_sentence))
                    self.batch_un_key_count.append(un_key_count)
                    self.batch_ct_words.append(ct_words)
                    self.batch_ct_labels.append(ct_labels)
                    self.batch_group.append(group_one_sent_f)

                    one_sentence = []
                    keys_one_sent = []
                    label_one_sent = []
                    one_src_sentence = []
                    pos_one_sent = []
                    ct_words = []
                    ct_labels = []
                    un_key_count = 0
                    src_count = 0

                    if args.dump_index>=0 and len(self.batch_keys)>args.dump_index:
                        break

        return tran_trg_filter_vocab

    def load_dict(self, dict_list, dict_pickle, tokenizer):
        print('load_dict...')
        if os.path.exists(dict_pickle):
            with open(dict_pickle, 'rb') as f:
                self.token_dict = pickle.load(f)
        else:
            '''
            # this maps source word to [best target, next target, next target...]
            self.wordmap = {} #defaultdict(list)

            # map punctuation so we have a good idea of how much is translated.
            # TODO: consider not counting punctuation.
            punc = list(u"<>/?:;()`~!@#$%^&*-_=+|[]{}.,»")
            otherpunc = ["...",u'",',u"”,",u"”.",u"»,",u"»."]
            for p in punc + otherpunc:
                self.wordmap[p].append(p)

            # be sure to order files in flist according to trustworthiness, most to least.
            # that is, if a word has multiple translations, the first is the best.
            # TODO: include scores in wordmap also
            '''

            for fname in dict_list:
                #with open(f, 'r') as f:
                with codecs.open(fname,"r","utf-8") as f:
                    #lines = f.readlines()
                    
                    #for line in lines:
                    #    sline = line.strip().split()
                    #    if len(sline) > 1:
                    #        self.wordmap[sline[0]].append(sline[1])

                    lan_dict = {}
                    for i, line in enumerate(f):
                        item = line.split(' ')
                        if len(item)<2:
                            continue
                        src_word = item[0].strip()
                        trg_word = item[1].replace('\n','').strip()
                        if len(src_word)==0 or len(trg_word)==0:
                            continue
                        #if src_word not in self.wordmap:
                        #    self.wordmap[src_word]=[]
                        #self.wordmap[src_word].append(trg_word)

                        if src_word not in lan_dict:
                            lan_dict[src_word] = []
                        if trg_word not in [itm[0] for itm in lan_dict[src_word]]:
                            p_tokens = tokenizer(' '+trg_word, return_tensors='pt', add_special_tokens=False)
                            lan_dict[src_word].append([trg_word,0.0,p_tokens['input_ids'][0].tolist()])

                        if case_sensitive:
                            if src_word.isupper() or src_word[0].isupper():
                                src_word = src_word.lower()
                                trg_word = trg_word.lower()
                                if src_word not in lan_dict:
                                    lan_dict[src_word] = []
                                if trg_word not in [itm[0] for itm in lan_dict[src_word]]:
                                    p_tokens = tokenizer(' '+trg_word, return_tensors='pt', add_special_tokens=False)
                                    lan_dict[src_word].append([trg_word,0.0,p_tokens['input_ids'][0].tolist()])

                            src_word = src_word[0].upper() + src_word[1:]
                            trg_word = trg_word[0].upper() + trg_word[1:]
                            if src_word not in lan_dict:
                                lan_dict[src_word] = []
                            if trg_word not in [itm[0] for itm in lan_dict[src_word]]:
                                p_tokens = tokenizer(' '+trg_word, return_tensors='pt', add_special_tokens=False)
                                lan_dict[src_word].append([trg_word,0.0,p_tokens['input_ids'][0].tolist()])

                            src_word = src_word.upper()
                            trg_word = trg_word.upper()
                            if src_word not in lan_dict:
                                lan_dict[src_word] = []
                            if trg_word not in [itm[0] for itm in lan_dict[src_word]]:
                                p_tokens = tokenizer(' '+trg_word, return_tensors='pt', add_special_tokens=False)
                                lan_dict[src_word].append([trg_word,0.0,p_tokens['input_ids'][0].tolist()])
                        print(i)
            with open(dict_pickle, 'wb') as f:
                pickle.dump(lan_dict, f)
            self.token_dict = lan_dict



    def __len__(self):
        if args.dump_index >= 0:
            return 1
        return len(self.batch_inp_sent)

    def __getitem__(self, idx):
        #return (self.batch_inp_sent[idx], self.batch_keys[idx])
        if args.dump_index >= 0:
            idx = args.dump_index
        print('%d/%d======================'%(idx, len(self.batch_inp_sent)))
        if log_f is not None:
            log_f.write('%d/%d======================'%(idx, len(self.batch_inp_sent))+'\n')
            #print(self.batch_src_sent[idx])
            #print(self.batch_ct_words[idx])
            log_f.write('src: ' + self.batch_src_sent[idx]+'\n')
            log_f.write('cheap tran: ' + ' '.join(self.batch_ct_words[idx]) +'\n')
            if 'gpt2' not in model_type:
                log_f.write('input: '+ self.batch_inp_sent[idx]+'\n')
            log_f.write('keys: '+ ','.join(['['+','.join([key for key in keys])+
                ' (%s)<%d>]'%(l[0][1:-1],g) for keys,l,g in zip(self.batch_keys[idx],self.batch_labels[idx],self.batch_group[idx])])+
                ' un_key_count:'+str(self.batch_un_key_count[idx])+'\n')
        else:
            print('src: ', self.batch_src_sent[idx])
            print('cheap tran: ', ' '.join(self.batch_ct_words[idx]))
            if 'gpt2' not in model_type:
                print('input: ', self.batch_inp_sent[idx])
            print('keys: ',self.batch_keys[idx])
            print('label: ',self.batch_labels[idx])
        if 'gpt2' in model_type:
            return ('<skip>' if (use_direct_tran or self.batch_inp_sent[idx]=='<skip>') else '.', self.batch_keys[idx], self.batch_labels[idx], self.batch_un_key_count[idx],
                self.batch_ct_words[idx], self.batch_ct_labels[idx], self.batch_group[idx])
        else:
            return ('<skip>' if use_direct_tran else self.batch_inp_sent[idx], self.batch_keys[idx], self.batch_labels[idx], self.batch_un_key_count[idx],
                self.batch_ct_words[idx], self.batch_ct_labels[idx], self.batch_group[idx])


    def get_tran_words(self, ph):
        #opt = self.wordmap.get(ph, [ph])
        #return [[w,0.0] for w in opt]
        opt = self.token_dict.get(ph,[[ph,0.0,[]]])
        return [[wi[0],0.0] for wi in opt ]

def dump(decoder, tokenizer, input_data, dump_f):
    print('dump ..')
    class my_logger():
        def __init__(self):
            self.tokenizer=tokenizer
        def add(self, log_text):
            if dump_f is None:
                print(log_text)
            else:
                dump_f.write(log_text+'\n')
        def get_tran_words(self, ph):
            return input_data.get_tran_words(ph)
    decoder.dump(my_logger())
    if dump_f is not None:
        dump_f.close()


if __name__ == '__main__':
    if ',' in args.gpu_index:
        dev_id = args.gpu_index.split(',')
        dev_list = [torch.device('cuda:'+ i if int(i)>=0 else 'cpu') for i in dev_id]
    elif int(args.gpu_index) < 0:
        dev_list = [torch.device('cpu')]
    else:
        dev_list = [torch.device('cuda:'+ args.gpu_index if torch.cuda.is_available() else 'cpu')]

    if len(input_filter_vocab) > 0:
        input_filter_vocab = pickle.load(open(input_filter_vocab, 'rb'))
    else:
        input_filter_vocab =None

    if len(key_filter_vocab) > 0:
        key_filter_vocab = pickle.load(open(key_filter_vocab, 'rb'))
    else:
        key_filter_vocab = None

    if args.pydev:
        import pydevd
        pydevd.settrace("localhost", port=5678) 

    model_list=[]
    model=None
    if 't5' in model_type:
        if config['model']['custom_tokenizer']:
            print('read source data, use t5 custom tokenizer')
            tokenizer = T5Tokenizer.from_pretrained(config['model']['custom_tokenizer'])
        else:
            print('read source data, use t5 pretrained tokenizer')
            tokenizer = T5Tokenizer.from_pretrained(model_type)

        if train_type == 'fine-tune':
            model = T5ForConditionalGeneration.from_pretrained(model_type, return_dict=True).to(dev_list[0])
        else:
            pad_token_id = tokenizer.pad_token_id
            model_spec = T5Config(
              decoder_start_token_id=pad_token_id
            ).from_pretrained(model_type) # creating the model

            model = T5ForConditionalGeneration(model_spec).to(dev_list[0])
    elif 'bart' in model_type:
        if config['model']['custom_tokenizer']:
            print('read source data, use bart custom tokenizer')
            tokenizer = BartTokenizer.from_pretrained(config['model']['custom_tokenizer'])
        else:
            print('read source data, use bart pretrained tokenizer')
            tokenizer = BartTokenizer.from_pretrained(model_type)

        if train_type == 'fine-tune':
            model = BartForConditionalGeneration.from_pretrained(model_type, return_dict=True).to(dev_list[0])
        else:
            model_spec = BartConfig().from_pretrained(model_type)
            model_list = []
            if len(dev_list)>1:
                for dev in dev_list:
                    model_list.append(BartForConditionalGeneration(model_spec).to(dev))
                    model_list[-1].eval()
                model = None
            else:
                model = BartForConditionalGeneration(model_spec).to(dev_list[0])
    elif 'gpt2' in model_type:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        model = GPT2LMHeadModel.from_pretrained(model_type).to(dev_list[0])
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if model_path is not None and len(model_path)>0:
        print("\n\nLoad the previous model from: " + model_path)
        print("\n\n")
        load_weigt=torch.load(model_path, map_location=torch.device('cpu'))
        if model is None:
            for model_i in model_list:
                model_i.load_state_dict(load_weigt)    
        else:
            model.load_state_dict(load_weigt)

    if model is not None:
        model.eval()

    if 'bart' in model_type:
        end_token = '</s>'
    else:
        end_token = ''  #<pad> is used as end_toekn in t5, so need not add it
    src_lines = []
    trg_lines = []
    key_lines = []

    if len(args.src_path) == 0:
        src_path = config['data'].get('default_src', '')

    if os.path.exists(src_path):
        if config['data']['input_type'] == 'CheapTranInput':
            input_data = CheapTranInputData(src_path, dict_list, input_filter_vocab=input_filter_vocab, tokenizer=tokenizer)
        else:
            input_data = SimpleTranInputData(src_path, dict_list, input_filter_vocab=input_filter_vocab, tokenizer=tokenizer)
    else:
        print('src path of %s does not exist!!!'%src_path)
        sys.exit()
        '''
        inp_en = '<mask> politics responded <mask> test <mask>'
        key_de = '<mask> politik <mask> sperre <mask>'
        input_data = TestInputData([inp_en], [key_de])
        '''
    print('input_data len:', len(input_data))
    filter_vocab = None
    if 'dict' in constrain_type:
        #if config['data']['input_type'] == 'CheapTranInput':
        token_dict = input_data.token_dict
        filter_vocab = input_data.filter_token_vocab
        #else:
        #    token_dict = load_dict(dict_list, dict_cache, tokenizer)
    elif 'hard' in constrain_type:
        token_dict = None
    else:
        print('constrain_type of %s not support!!!'%constrain_type)
        sys.exit()

    if args.print_log is None:
        print_log_f = None
    else:
        if len(args.print_log)==0:
            print_log_f = sys.stdout
        else:
            try:
                print_log_f = open(args.print_log, 'w')
                print('use log file: %s'%args.print_log)
            except:
                print_log_f = sys.stdout

    #use_cache = len(model_list)>1 or (int(args.gpu_index)>=0 and args.process_size==-1)
    decoder = GridBeamSearchDecoder(model, tokenizer, beam_size=num_beams, constrain_type=constrain_type,
        lan_dict=token_dict, filter_vocab=filter_vocab,
        process_size=args.process_size, use_cache=args.use_cache, model_list=model_list,
        high_gpu_mem_th=args.high_mem, low_gpu_mem_th=args.low_mem, mon_gpu_mem_idx=int(args.gpu_index),
        debug_flag=args.dump_index>=0 or args.debug, print_log=print_log_f)
    
    #print('do_sample:', do_sample)
    def save_output_fun(pred, data=None, idx=0, score=0):
        global tot_score, score_count
        output_file = output_file0 if idx==0 else output_file0+str(idx)
        if reverse_order:
            rev_output_file = rev_output_file0 if idx==0 else rev_output_file0+str(idx)
        if pred is None:
            if args.dump_index<0:
                with open(output_file, "a") as f:
                    for w,l in zip(data[4], data[5]): #self.batch_ct_words[idx], self.batch_ct_labels[idx]
                        f.write('%s %s %s %s\n'%(w, w, l, l))
                    f.write('\n')

                if reverse_order:
                    with open(rev_output_file, "a") as f:
                        for w,l in zip(data[4], data[5]): #self.batch_ct_words[idx], self.batch_ct_labels[idx]
                            f.write('%s %s %s %s\n'%(w, w, l, l))
                        f.write('\n')

            if log_f is not None:
                if data[0]=='<skip>':
                    log_f.write('Use CT\n') 
                else:
                    log_f.write('No done-hyps\n') 
                    if False:
                        log_f.flush()
                        print('wwwwwwwwwwwwwwww')
                        dump_file = output_file0+'.dump'
                        dump_f=open(dump_file, 'w')
                        dump(decoder, tokenizer, input_data, dump_f)
                        sys.exit()
            return

        if log_f is not None:
            log_f.write('predicted with label: '+pred+'\n') 
        else:      
            print('predicted with label: ',pred)
        words = [w for w in pred.split(' ') if len(w)>0]
        word_list = []
        label_list = []
        label = 'O'
        count = 0
        for w in words:
            w = w.strip()
            if w in input_data.labels or w==input_data.un_labeled:
                label = w[1:-1]
                count = 0
            else:
                word_list.append(w)
                if label!='O':
                    if count==0:
                        if len(label_list)>0:
                            label_list[-1]=label_list[-1].replace('B-','I-')
                        label_list.append('B-'+label)
                    else:
                        label_list.append('I-'+label)
                else:
                    if len(label_list)>0:
                        label_list[-1]=label_list[-1].replace('B-','I-')
                    label_list.append(label)
                count+=1

        if args.dump_index<0:
            with open(output_file, "a") as f:
                for w,l in zip(word_list, label_list):
                    if len(w)>0:
                        f.write('%s %s %s %s\n'%(w, w, l, l))
                f.write('\n')
        else:
            for w,l in zip(word_list, label_list):
                if len(w)>0:
                    print('%s %s %s %s\n'%(w, w, l, l))

        tot_score+=score
        score_count+=1
        if log_f is not None:
            pred_sent=' '.join(word_list)
            #print('score:', score)
            log_f.write('predicted(%f): %s\n'%(score,pred_sent)) 
            log_f.flush()
            if args.debug and len(pred_sent.strip())==0:
                print('wwwwwwwwwwwwwwww')
                dump_file = output_file0+'.dump'
                dump_f=open(dump_file, 'w')
                dump(decoder, tokenizer, input_data, dump_f)
                sys.exit()
        else:
            print('predicted: ',' '.join(word_list))

        if reverse_order and args.dump_index<0:
            with open(rev_output_file, "a") as f:
                word_list = word_list[::-1]
                label_list = label_list[::-1]
                for w,l in zip(word_list, label_list):
                    if len(w)>0:
                        f.write('%s %s %s %s\n'%(w, w, l, l))
                f.write('\n')

    if do_sample:
        hypotheses, all_preds = decoder.decode(input_data=input_data, 
            output_fn=save_output_fun if len(output_file0)>0 else None, 
            n_best=args.n_best,
            start_index=args.start_index,
            policy=policy,
            max_unconstraint_len=max_unconstraint_len,
            min_len = 5,
            unkey_uc_ratio=unkey_uc_ratio,
            do_sample=True, repetition_penalty=repetition_penalty)
    else:
        hypotheses, all_preds = decoder.decode(input_data=input_data,
            output_fn=save_output_fun if len(output_file0)>0 else None,
            n_best=args.n_best,
            start_index=args.start_index,
            policy=policy,
            max_unconstraint_len=max_unconstraint_len,
            unkey_uc_ratio=unkey_uc_ratio,
            no_repeat_ngram_size=no_repeat_ngram_size)
    
    if log_f is not None and score_count>0:
        log_f.write('decoded count %d tot_score %f  avg %f\n'%(score_count, tot_score, tot_score/score_count)) 
        log_f.flush()

    for p in all_preds:
        print('predicted: ', p)

    if args.dump_index>=0:
        dump(decoder, tokenizer, input_data, dump_f)
        '''
        print('dump ..')
        class my_logger():
            def __init__(self):
                self.tokenizer=tokenizer
            def add(self, log_text):
                if dump_f is None:
                    print(log_text)
                else:
                    dump_f.write(log_text+'\n')
            def get_tran_words(self, ph):
                return input_data.get_tran_words(ph)
        decoder.dump(my_logger())
        if dump_f is not None:
            dump_f.close()
        '''




