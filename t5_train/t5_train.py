#!/u/subramas/miniconda2/bin/python
"""Main script to run things"""
import sys

sys.path.append("../transformers/src/")

from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
import numpy as np
import logging
import argparse
import os
import pickle

import torch

from transformers.modeling_t5 import T5ForConditionalGeneration, T5Config
from transformers.modeling_bart import BartForConditionalGeneration, BartConfig

from transformers.optimization import AdamW

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--model', type=str, default='')
parser.add_argument('--type', type=str, default='train')
parser.add_argument('--test_batch_size', type=int, default=0)
parser.add_argument('--process_size', type=int, default=32)
parser.add_argument('--gps_batch_process', type=bool, default=False)
parser.add_argument('--decode_max_len', type=int, default=36)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument('--do_sample', type=bool, default=False)
parser.add_argument('--pydev', type=bool, default=False, help='enable pydev debug.')
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--print_log', type=bool, default=False)

args = parser.parse_args()

if args.gpu_index < 0:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+ str(args.gpu_index) if torch.cuda.is_available() else 'cpu')

config_file_path = args.config
config = read_config(config_file_path)
config['training']['device'] = device
experiment_name = hyperparam_string(config)

train_type = config['training']['type']
model_type = config['model']['type']
data_type = config['data']['type']
save_dir = config['data']['save_dir']
noise = config['training']['noise']
if noise > 0.0:
    noise_vocab = pickle.load(open(config['training']['noise_vocab'], 'rb'))

def get_weights_save_path(epoch, save=False):
    path_with_noise = os.path.join(
            save_dir,
            model_type.replace('/','-') + '_' + data_type + '_noise' + str(noise) +'_epoch_%d' % (epoch) + '.ckpt')
    if noise > 0.0  and (save or os.path.exists(path_with_noise)):
        path = path_with_noise
    else:
        path = os.path.join(
            save_dir,
            model_type.replace('/','-') + '_' + data_type + '_epoch_%d' % (epoch) + '.ckpt')
    return path

if args.load_epoch>=0:
    load_path = get_weights_save_path(args.load_epoch)
elif len(args.model)>0:
    load_path = args.model
else:
    load_path = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

if args.pydev:
    import pydevd
    pydevd.settrace("localhost", port=5678) 
# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


print('Reading data ...')

if 'train' in args.type:
    src, trg = read_nmt_data(
        src=config['data']['src'],
        config=config,
        trg=config['data']['trg']
    )

if 'test' in args.type or 'eval' in args.type:
    src_test, trg_test = read_nmt_data(
        src=config['data']['test_src'],
        config=config,
        trg=config['data']['test_trg']
    )

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']

if 't5' in model_type:
    if train_type == 'fine-tune':
        model = T5ForConditionalGeneration.from_pretrained(model_type, return_dict=True).to(device)
    else:
        if 'train' in args.type:
            pad_token_id = trg['tokenizer'].pad_token_id
        else:
            pad_token_id = trg_test['tokenizer'].pad_token_id
        model_spec = T5Config(
          decoder_start_token_id=pad_token_id
        ).from_pretrained(model_type) # creating the model

        model = T5ForConditionalGeneration(model_spec).to(device)
elif 'bart' in model_type:
    if train_type == 'fine-tune':
        model = BartForConditionalGeneration.from_pretrained(model_type, return_dict=True).to(device)
    else:
        model_spec = BartConfig().from_pretrained(model_type)
        model = BartForConditionalGeneration(model_spec).to(device)

if load_path is not None:
    print("\n\nLoad the previous model from: " + load_path)
    print("\n\n")
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

def test(input_lines_src, input_lines_trg):
    #model.eval()
    tokenizer = src_test['tokenizer']

    outputs = model(input_ids=input_lines_src, labels=input_lines_trg)
    scores, attn = outputs[:2] 
    print(scores)
    outputs = model.generate(input_lines_src, max_length=max_length, num_beams=args.num_beams, do_sample=True)
    for i,(k,l) in enumerate(zip(input_lines_src, input_lines_trg)):
        #predicted_token = tokenizer.convert_ids_to_tokens(outputs[0])
        #print('gold: ', tokenizer.convert_ids_to_tokens(l))
        #print('predicted: ', predicted_token)
        print('=========================')
        print('key: ', tokenizer.decode(k))
        print('gold: ', tokenizer.decode(l))
        print('predicted: ', tokenizer.decode(outputs[i]))    


if 'train' in args.type:
    model.train()

    lr = config['training']['lrate']
    optimizer = AdamW(model.parameters(), lr=lr)

    for i in range(args.load_epoch+1, 1000):
        losses = []
        print(src['size'])
        if i > 0 and noise > 0:
            src, _ = read_nmt_data(
                src=config['data']['src'],
                config=config,
                noise_vocab = noise_vocab
            )
        for j in range(0, src['size'], batch_size):

            input_lines_src, _, lens_src, mask_src = get_minibatch(
                src['data'], src['word2id'], j,
                batch_size, max_length, add_start=True, add_end=True
            )
            input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
                trg['data'], trg['word2id'], j,
                batch_size, max_length, add_start=True, add_end=True
            )

            model.zero_grad()
            outputs = model(input_ids=input_lines_src, labels=input_lines_trg)

            loss = outputs[0]
            losses.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

            if j % config['management']['monitor_loss'] == 0:
                logging.info(' ================    monitor_loss    ================')
                logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (
                    i, j, np.mean(losses))
                )
                losses = []


        if i % config['management']['checkpoint_freq'] == 0:
            logging.info(" ===================   Saving model after complete epochs  ===================")
            torch.save(
                model.state_dict(),
                get_weights_save_path(i, save=True)
            )
            if 'eval' in args.type:
                print('Training Set Eval:')
                input_lines_src, _, lens_src, mask_src = get_minibatch(
                    src['data'], src['word2id'], 0,
                    4, max_length, add_start=True, add_end=True
                )
                input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
                    trg['data'], trg['word2id'], 0,
                    4, max_length, add_start=True, add_end=True
                )
                test(input_lines_src, input_lines_trg)

                print('Test Set Eval:')
                input_lines_src, _, lens_src, mask_src = get_minibatch(
                    src_test['data'], src_test['word2id'], 0,
                    4, max_length, add_start=True, add_end=True
                )
                input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
                    trg_test['data'], trg_test['word2id'], 0,
                    4, max_length, add_start=True, add_end=True
                )
                test(input_lines_src, input_lines_trg)

elif args.type =='test':
    model.eval()
    if args.test_batch_size > 0:
        batch_size = args.test_batch_size

    input_lines_src, _, lens_src, mask_src = get_minibatch(
        src_test['data'], src_test['word2id'], 0,
        batch_size, max_length, add_start=True, add_end=True
    )
    input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
        trg_test['data'], trg_test['word2id'], 0,
        batch_size, max_length, add_start=True, add_end=True
    )

    test(input_lines_src, input_lines_trg)
elif args.type == 'test_grid_beam':
    sys.path.append("../grid_beam_search")
    from new_decode_gbs import GridBeamSearchDecoder

    model.eval()

    if args.test_batch_size > 0:
        batch_size = args.test_batch_size

    src=config['data']['test_src']
    trg=config['data']['test_trg']
    key_file=config['data']['test_key']

    if 'bart' in model_type:
        end_token = '</s>'
    else:
        end_token = ''  #<pad> is used as end_toekn in t5, so need not add it
    src_lines = []
    trg_lines = []
    key_lines = []
    print(src)
    with open(src, 'r') as f:
        for ind, line in enumerate(f):
            src_lines.append(line+end_token)

    with open(trg, 'r') as f:
        for ind, line in enumerate(f):
            trg_lines.append(line+end_token)

    with open(key_file, 'r') as f:
        for ind, line in enumerate(f):
            key_lines.append(line)

    tokenizer = src_test['tokenizer']
    decoder = GridBeamSearchDecoder(model, tokenizer, beam_size=args.num_beams, process_size=args.process_size, print_log=args.print_log)

    if args.gps_batch_process:
        if args.do_sample:
            hypotheses, all_preds = decoder.decode(src_lines[:batch_size], key_lines[:batch_size], do_sample=True, repetition_penalty=1.5)
        else:
            hypotheses, all_preds = decoder.decode(src_lines[:batch_size], key_lines[:batch_size], no_repeat_ngram_size=1)

        for i,(inp,lab,keys) in enumerate(zip(src_lines[:batch_size], trg_lines[:batch_size], key_lines[:batch_size])):
            print('=========================')
            print('inp: ', inp)
            print('gold: ', lab)
            print('keys: ', keys)
            print('predicted: ', all_preds[i])    

    else:
        for inp, lab, keys in zip(src_lines[:batch_size], trg_lines[:batch_size], key_lines[:batch_size]):
            if args.do_sample:
                hypotheses, all_preds = decoder.decode([inp], [keys], max_length=args.decode_max_len, do_sample=True, repetition_penalty=1.5)
            else:
                hypotheses, all_preds = decoder.decode([inp], [keys], max_length=args.decode_max_len, no_repeat_ngram_size=1)
            print('=========================')
            print('inp: ', inp)
            print('gold: ', lab)
            print('keys: ', keys)
            print('predicted: ', all_preds[0])    
