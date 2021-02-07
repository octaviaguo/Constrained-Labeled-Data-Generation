"""Grid Beam Search"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pickle
from typing import Iterable, List, Optional, Tuple
from torch import Tensor
from grid_beam import GridBeamSearch, Key
from transformers.modeling_outputs import BaseModelOutput
from threading import Thread
from pynvml import *

class InputData(object):
    def __init__(self, batch_sent, batch_key_phrases):
        self.batch_sent = batch_sent
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
            self.batch_keys = [None]*len(self.batch_sent)

    def __len__(self):
        return len(self.batch_sent)

    def __getitem__(self, idx):
        return (self.batch_sent[idx], self.batch_keys[idx])


class GridBeamSearchDecoder(object):
    """Beam Search decoder."""
    '''
    def bart_reorder_buffer(self, attn_cache, new_order):
        attn_cache_new = {}
        for k, input_buffer_k in attn_cache.items():
            if input_buffer_k is not None:
                attn_cache_new[k] = input_buffer_k.index_select(0, new_order)
                attn_cache[k] = None
                del input_buffer_k
        del attn_cache
        return attn_cache_new

    def bart_reorder_cache(self, past, indice):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: self.bart_reorder_buffer(attn_cache, indice) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
            del layer_past
        del past
        return reordered_past
    '''

    def bart_get_buffer_cpu(self, attn_cache):
        #attn_cache_new = {}
        for k, input_buffer_k in attn_cache.items():
            if input_buffer_k is not None:
                attn_cache[k] = input_buffer_k.cpu()
                del input_buffer_k
        return attn_cache

    def bart_get_cache_cpu(self, past):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: self.bart_get_buffer_cpu(attn_cache) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def bart_get_buffer(self, attn_cache, indice):
        attn_cache_new = {}
        for k, input_buffer_k in attn_cache.items():
            if input_buffer_k is not None:
                attn_cache_new[k] = input_buffer_k.index_select(0, indice).to(self.model.device)
        return attn_cache_new

    def bart_get_cache(self, past, indice):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: self.bart_get_buffer(attn_cache, indice) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def bart_get_buffer_dev(self, attn_cache, indice, dev):
        attn_cache_new = {}
        for k, input_buffer_k in attn_cache.items():
            if input_buffer_k is not None:
                attn_cache_new[k] = input_buffer_k.index_select(0, indice).to(dev)
        return attn_cache_new

    def bart_get_cache_dev(self, past, indice, dev): #same as bart_get_cache except dev
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: self.bart_get_buffer_dev(attn_cache, indice, dev) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def bart_concat_buffer(self, attn_cache, attn_cache_list):
        attn_cache_new = {}
        for k, input_buffer_k in attn_cache.items():
            if input_buffer_k is not None:
                buffer_list = [attn_cache_list[i][k].cpu() for i in range(len(attn_cache_list))]
                attn_cache_new[k] = torch.cat(buffer_list)
        return attn_cache_new

    def bart_concat_cache(self, past_list_):
        concated_past = []
        for l,layer_past in enumerate(past_list_[0]):
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: self.bart_concat_buffer(attn_cache, [past_list_[i][l][attn_key] for i in range(len(past_list_))]) for attn_key, attn_cache in layer_past.items()
            }
            concated_past.append(layer_past_new)
        return concated_past

    def reorder_cache(self, past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)

    def get_cache(self, past, indice):
        return tuple(layer_past.index_select(1, indice).to(self.model.device) for layer_past in past)

    def get_cache_cpu(self, past):
        ret = tuple(layer_past.cpu() for layer_past in past)
        for layer_past in past:
            del layer_past
        return ret

    def concat_cache(self, past_list_):
        concated_past = []
        for l,layer_past_list in enumerate(zip(*past_list_)):
            layer_past_new = torch.cat([buf.cpu() for buf in layer_past_list], dim=1)
            concated_past.append(layer_past_new)
        return concated_past

    def __init__(
        self,
        model,
        tokenizer,
        beam_size=1,
        model_type = 'bart',
        constrain_type='hard',
        lan_dict=None,
        filter_vocab=None,
        model_list=[],
        process_size=32,
        use_cache=True,
        high_gpu_mem_th=0,
        low_gpu_mem_th=0,
        mon_gpu_mem_idx=0,
        debug_flag=False,
        print_log=None,
    ):
        #high_gpu_mem_th=2000000000
        #low_gpu_mem_th=1000000000
        if high_gpu_mem_th > 0:
            nvmlInit()
        self.use_sub_batch_decode = False
        self.batch_decode_count = 0
        self.sub_batch_decode_count = 0
        self.used_gpu_memory_max = 0
        self.beam_size = beam_size
        self.constrain_type=constrain_type
        if model is None:
            self.model = model_list[0]
        else:
            self.model = model
        self.model_list = model_list
        self.tokenizer = tokenizer
        self.past = None
        self.dict = lan_dict
        self.filter_vocab = filter_vocab
        self.trg_token_dict = {}
        self.resource = {'trg_token_dict':self.trg_token_dict, 'dict':self.dict, 'filter_vocab':self.filter_vocab}
        if self.model.config.is_encoder_decoder: #'bart' in self.model_type:
            self._concat_cache = self.bart_concat_cache
            self._get_cache = self.bart_get_cache_dev if len(model_list)>1 else self.bart_get_cache
            self._get_cache_cpu = self.bart_get_cache_cpu
            if len(model_list)<=1:
                self._reorder_cache = model._reorder_cache
        else:
            self._concat_cache = self.concat_cache
            self._get_cache = self.get_cache
            self._get_cache_cpu = self.get_cache_cpu
            self._reorder_cache = self.reorder_cache

        def batch_decode(input, states, t):
            input_ids = states[0]
            pre_hyp_index = states[-1]
            if t>1:
                input_ids = torch.cat([input_ids, input.to(self.model.device)], dim=-1)
                past = self._reorder_cache(self.past, pre_hyp_index.squeeze(-1).to(self.model.device))
            else:
                past = None

            if past is None:
                model_input_ids = input_ids.to(self.model.device)
            else:
                model_input_ids = input_ids[:, -1].unsqueeze(-1).to(self.model.device)

            model_inputs = {
                "input_ids": model_input_ids,  
                "past_key_values": past,
                "use_cache": True,  # change this to avoid caching (presumably for debugging)
            }

            outputs = self.model(**model_inputs, return_dict=True)

            self.past = outputs.past_key_values
            cur_hyp_index = torch.tensor([i for i in range(input_ids.shape[0])]).view(input_ids.shape[0], 1)

            ret_states = (input_ids, cur_hyp_index)
            return outputs.logits, ret_states


        def batch_decode_bart(input, states, t, past_in_cpu=False):
            input_ids = states[0].to(self.model.device)
            last_hidden_state = states[1].to(self.model.device)
            attention_mask = states[2].to(self.model.device)
            pre_hyp_index = states[3]
            if t>1:
                input_ids = torch.cat([input_ids, input.to(self.model.device)], dim=-1)
                if not use_cache:
                    past = None
                else:
                    if past_in_cpu:
                        past = self._get_cache(self.past, pre_hyp_index.squeeze(-1))
                    else:
                        past = self.model._reorder_cache(self.past, pre_hyp_index.squeeze(-1).to(self.model.device))
                    '''
                    print('begin3:',torch.cuda.memory_reserved())
                    past = self.bart_reorder_cache(self.past, pre_hyp_index.squeeze(-1).to(self.model.device))
                    del self.past
                    self.past = None
                    print('before:',torch.cuda.memory_reserved())
                    torch.cuda.empty_cache()
                    print('after:',torch.cuda.memory_reserved())
                    '''
                '''
                else:
                    print('before:',torch.cuda.memory_reserved())
                    torch.cuda.empty_cache()
                    print('after:',torch.cuda.memory_reserved())

                    past = self.model._reorder_cache(self.past, pre_hyp_index.squeeze(-1).cpu())
                    past_index = torch.tensor([i for i in range(states[0].shape[0])])
                    past = self.bart_get_cache(past, past_index)
                
                print('cuda used memory:',torch.cuda.memory_reserved())
                if torch.cuda.memory_reserved()>805306368:
                    past = None
                '''
            else:
                past = None
            encoder_outputs = BaseModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=None,
                attentions=None
            )

            model_inputs = {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }

            outputs = self.model(**model_inputs, return_dict=True)

            #if True:
            self.past = outputs.past_key_values
            '''
            else:
                self.past = self.bart_concat_cache([self.bart_get_cache_cpu(outputs.past_key_values)])
                if past is not None:
                    self.bart_get_cache_cpu(past)
            '''
            cur_hyp_index = torch.tensor([i for i in range(input_ids.shape[0])]).view(input_ids.shape[0], 1).to(self.model.device)
            return outputs.logits, (input_ids, outputs.encoder_last_hidden_state, attention_mask, cur_hyp_index)

        '''
        sub batch decoding to avoid GPU running out of memory 
        '''
        def sub_batch_decode(input, states, t, past_in_gpu=False):
            with torch.no_grad():
                #use_cache = True #False only for debugging
                if past_in_gpu:
                    input_ids = states[0].cpu()
                    pre_hyp_index = states[-1].cpu()
                    if self.model.config.is_encoder_decoder:
                        last_hidden_state = states[1].cpu()
                        attention_mask = states[2].cpu()
                    if t>1:
                        input_ids = torch.cat([input_ids.cpu(), input], dim=-1)
                        if use_cache:
                            past = self._get_cache_cpu(self.past)
                            past = self._reorder_cache(past, pre_hyp_index.squeeze(-1))
                        else:
                            past = None
                    else:
                        past = None
                else:
                    input_ids = states[0]
                    pre_hyp_index = states[-1]
                    if self.model.config.is_encoder_decoder:
                        last_hidden_state = states[1]
                        attention_mask = states[2]
                    if t>1:
                        input_ids = torch.cat([input_ids, input], dim=-1)
                        if use_cache:
                            past = self._reorder_cache(self.past, pre_hyp_index.squeeze(-1))
                        else:
                            past = None
                    else:
                        past = None

                logits_list = []
                hidden_state_list = []
                past_list = []
                for i in range(0, states[0].shape[0], process_size):
                    if print_log:
                        print('=============before decoding', file=print_log)
                        #print(torch.cuda.memory_reserved())

                    if t>1 and use_cache:
                        past_index = torch.tensor([i+j for j in range(process_size) if (i+j)<states[0].shape[0]])
                        past_i = self._get_cache(past, past_index)
                    else:
                        past_i = None

                    if self.model.config.is_encoder_decoder:
                        last_hidden_state_ = last_hidden_state[i:i+process_size].to(self.model.device)
                        input_ids_ = input_ids[i:i+process_size].to(self.model.device)
                        attention_mask_ = attention_mask[i:i+process_size].to(self.model.device)
                        encoder_outputs = BaseModelOutput(
                            last_hidden_state=last_hidden_state_,
                            hidden_states=None,
                            attentions=None
                        )

                        model_inputs = {
                            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                            "encoder_outputs": encoder_outputs,
                            "past_key_values": past_i,
                            "decoder_input_ids": input_ids_,
                            "attention_mask": attention_mask_,
                            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
                        }
                    else:
                        if past_i is None:
                            input_ids_ = input_ids[i:i+process_size].to(self.model.device)
                        else:
                            input_ids_ = input_ids[i:i+process_size, -1].unsqueeze(-1).to(self.model.device)

                        model_inputs = {
                            "input_ids": input_ids_,  
                            "past_key_values": past_i,
                            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
                        }

                    outputs = self.model(**model_inputs, return_dict=True)
                    #print(outputs)
                    logits_list.append(outputs.logits.cpu())
                    if self.model.config.is_encoder_decoder:
                        hidden_state_list.append(outputs.encoder_last_hidden_state.cpu())

                    del outputs.logits
                    del input_ids_
                    if self.model.config.is_encoder_decoder:
                        del outputs.encoder_last_hidden_state
                        del last_hidden_state_
                        del attention_mask_

                    if use_cache:
                        past_list.append(self._get_cache_cpu(outputs.past_key_values))
                        if past_i is not None:
                            self._get_cache_cpu(past_i) # call this function to delete cuda tensor
                    torch.cuda.empty_cache()
                    if print_log:
                        print('===============after decoding', file=print_log)
                        #print(torch.cuda.memory_reserved())
                logits = torch.cat(logits_list)
                if use_cache:
                    self.past = self._concat_cache(past_list)
                cur_hyp_index = torch.tensor([i for i in range(input_ids.shape[0])]).view(input_ids.shape[0], 1)
                
                if self.model.config.is_encoder_decoder:
                    encoder_last_hidden_state = torch.cat(hidden_state_list)
                    ret_states = (input_ids.cpu(), encoder_last_hidden_state, attention_mask, cur_hyp_index)
                else:
                    ret_states = (input_ids.cpu(), cur_hyp_index)
            return logits, ret_states

        def batch_decode_switch(input, states, t):
            #used_gpu_memory = torch.cuda.memory_reserved()
            h = nvmlDeviceGetHandleByIndex(mon_gpu_mem_idx)
            info = nvmlDeviceGetMemoryInfo(h)
            used_gpu_memory = info.used
            if used_gpu_memory>self.used_gpu_memory_max:
                self.used_gpu_memory_max=used_gpu_memory
            #print('used_gpu_memory:',used_gpu_memory)
            use_sub_batch_decode_pre = self.use_sub_batch_decode
            if used_gpu_memory > high_gpu_mem_th:
                self.use_sub_batch_decode = True

            if used_gpu_memory < low_gpu_mem_th:
                self.use_sub_batch_decode = False

            if self.use_sub_batch_decode:
                ret = sub_batch_decode(input, states, t, past_in_gpu = not use_sub_batch_decode_pre)
                self.sub_batch_decode_count+=1
            else:
                ret = batch_decode_bart(input, states, t, past_in_cpu = use_sub_batch_decode_pre)
                self.batch_decode_count+=1
            return ret


        def sub_batch_decode_multi_gpu(input, states, t):
            with torch.no_grad():
                #use_cache = True #False only for debugging
                input_ids = states[0]
                pre_hyp_index = states[-1]
                if self.model.config.is_encoder_decoder:
                    last_hidden_state = states[1]
                    attention_mask = states[2]
                if t>1:
                    input_ids = torch.cat([input_ids, input], dim=-1)
                    if use_cache:
                        past = self.model._reorder_cache(self.past, pre_hyp_index.squeeze(-1))
                    else:
                        past = None
                else:
                    past = None

                logits_list = []
                hidden_state_list = []
                past_list = []

                def model_decode_dev(proc_args, model):
                    for i in proc_args[0]: #i_list:
                        if print_log:
                            print('=============before decoding', file=print_log)
                            #print(torch.cuda.memory_reserved())

                        if t>1 and use_cache:
                            past_index = torch.tensor([i+j for j in range(process_size) if (i+j)<states[0].shape[0]])
                            past_i = self._get_cache(past, past_index, model.device)
                        else:
                            past_i = None

                        if model.config.is_encoder_decoder:
                            last_hidden_state_ = last_hidden_state[i:i+process_size].to(model.device)
                            input_ids_ = input_ids[i:i+process_size].to(model.device)
                            attention_mask_ = attention_mask[i:i+process_size].to(model.device)
                            encoder_outputs = BaseModelOutput(
                                last_hidden_state=last_hidden_state_,
                                hidden_states=None,
                                attentions=None
                            )

                            model_inputs = {
                                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                                "encoder_outputs": encoder_outputs,
                                "past_key_values": past_i,
                                "decoder_input_ids": input_ids_,
                                "attention_mask": attention_mask_,
                                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
                            }
                        else:
                            if past_i is None:
                                input_ids_ = input_ids[i:i+process_size].to(model.device)
                            else:
                                input_ids_ = input_ids[i:i+process_size, -1].unsqueeze(-1).to(model.device)

                            model_inputs = {
                                "input_ids": input_ids_,  
                                "past_key_values": past_i,
                                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
                            }

                        outputs = model(**model_inputs, return_dict=True)
                        #print(outputs)
                        proc_args[1].append(outputs.logits.cpu())
                        if model.config.is_encoder_decoder:
                            proc_args[2].append(outputs.encoder_last_hidden_state.cpu())

                        del outputs.logits
                        del input_ids_
                        if model.config.is_encoder_decoder:
                            del outputs.encoder_last_hidden_state
                            del last_hidden_state_
                            del attention_mask_

                        if use_cache:
                            proc_args[3].append(self._get_cache_cpu(outputs.past_key_values))
                            if past_i is not None:
                                self._get_cache_cpu(past_i) # call this function to delete cuda tensor
                        if print_log:
                            print('===============after decoding', file=print_log)
                            #print(torch.cuda.memory_reserved())

                process_dict = {}
                model_num = len(model_list)
                task_list = []
                for j in range(model_num):
                    process_dict[j]=[[],[],[],[],0] #i,logit,state,past
                j=0
                for i in range(0, states[0].shape[0], process_size):
                    process_dict[j][0].append(i)
                    j=j+1
                    if j>=model_num:
                        j=0
                for j in range(model_num):
                    if len(process_dict[j][0])>0:
                        task_list.append(Thread(target=model_decode_dev, args=(process_dict[j], model_list[j])))
                        task_list[-1].start()
                for tsk in task_list:
                    tsk.join()
                torch.cuda.empty_cache()
                j=0
                for i in range(0, states[0].shape[0], process_size):
                    logits_list.append(process_dict[j][1][process_dict[j][4]])
                    hidden_state_list.append(process_dict[j][2][process_dict[j][4]])
                    if use_cache:
                        past_list.append(process_dict[j][3][process_dict[j][4]])
                    process_dict[j][4]+=1
                    j=j+1
                    if j>=model_num:
                        j=0

                logits = torch.cat(logits_list) #proc_args[1]
                if use_cache:
                    self.past = self._concat_cache(past_list) #proc_args[3]
                cur_hyp_index = torch.tensor([i for i in range(input_ids.shape[0])]).view(input_ids.shape[0], 1)
                
                if self.model.config.is_encoder_decoder:
                    encoder_last_hidden_state = torch.cat(hidden_state_list) #proc_args[2]
                    ret_states = (input_ids.cpu(), encoder_last_hidden_state, states[2], cur_hyp_index)
                else:
                    ret_states = (input_ids.cpu(), cur_hyp_index)
            return logits, ret_states

        bos = self.model.config.decoder_start_token_id
        if bos is None:
            bos = self.tokenizer.bos_token_id

        if high_gpu_mem_th>0:
            decode_fun = batch_decode_switch
        elif process_size>0:
            if len(model_list)>1:
                decode_fun = sub_batch_decode_multi_gpu
            else:
                decode_fun = sub_batch_decode
        elif self.model.config.is_encoder_decoder: #'bart' in self.model_type:
            decode_fun = batch_decode_bart
        else:
            decode_fun = batch_decode

        self.grid_beam = GridBeamSearch(
            beam_size=self.beam_size,
            decode_fn=decode_fun,
            bos = bos,
            eos = self.tokenizer.eos_token_id,
            vocab_size=self.model.config.vocab_size, 
            constrain_type=self.constrain_type,
            debug_flag=debug_flag,
            print_log=print_log)


    def get_hidden_representation(self, input_ids):
        """Get hidden representation for a sentence."""
        if len(self.model_list)>1:
            for model in self.model_list:
                encoder = model.get_encoder()
                encoder_output = encoder(input_ids['input_ids'].to(model.device), attention_mask=input_ids['attention_mask'].to(model.device), return_dict=True)
        else:
            #attention_mask = input_ids.new_ones(input_ids.shape)
            encoder = self.model.get_encoder()
            encoder_output = encoder(input_ids['input_ids'].to(self.model.device), attention_mask=input_ids['attention_mask'].to(self.model.device), return_dict=True)

        return encoder_output

    def __add_dict(self, keys, labels, not_translate_labels):
        for p,l in zip(keys,labels):
            for w in p: 
                if (w not in self.dict) or (l[0] in not_translate_labels and w not in self.trg_token_dict.keys()):
                    p_tokens = self.tokenizer(' '+w, return_tensors='pt', add_special_tokens=False)
                if w not in self.dict:
                    self.dict[w] = [[w, 0.0, p_tokens['input_ids'][0].tolist()]]
                if l[0] in not_translate_labels and w not in self.trg_token_dict.keys():
                    self.trg_token_dict[w]=p_tokens['input_ids'][0].tolist()
                    #print('w:', self.trg_token_dict[w]) 

    def dump(self, logger):
        self.grid_beam.dump(logger)

    def decode(self, 
        batch_sent = None, 
        batch_key_phrases= None,
        input_data = None,
        output_fn = None,
        n_best = 1,
        start_index = 0,
        max_unconstraint_len=20,
        min_len=0,
        unkey_uc_ratio=0.0,
        decode_token=True,
        policy={},
        repetition_penalty=1.0,
        bad_words_ids=None,
        no_repeat_ngram_size=0,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0
        ):
        with_label = False
        ret_preds = []
        ret_hyp = []
        max_grp_uncs_len = policy.get('max_grp_uncs_len',0)
        print('max_grp_uncs_len is ', max_grp_uncs_len)
        if input_data is None:
            input_data = InputData(batch_sent, batch_key_phrases)

        if len(input_data[0])>=3:
            with_label = True
            for tok in input_data.labels + [input_data.un_labeled]:
                enc = self.tokenizer.encode(tok)
                #if 2 in enc:
                self.tokenizer.add_tokens(tok)
            label_dict = {}
            for tok in input_data.labels + [input_data.un_labeled]:
                tt = self.tokenizer(tok, max_length=80, return_tensors='pt', add_special_tokens=False)
                #aa = tokenizer.decode(hh['input_ids'][0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                label_dict[tok] = int(tt['input_ids'][0][0].numpy())
            #print(label_dict)
        batch_size = 1
        for i in range(len(input_data)):
            if i < start_index:
                continue
            d = input_data[i]
            if d[0]=='<skip>':
                if decode_token and output_fn:
                    for ih in range(min(n_best, self.beam_size)):
                        output_fn(None, data=d, idx=ih)
                continue
            input_lines_src = self.tokenizer([d[0]], max_length=80, return_tensors='pt', padding=True, truncation=True)
            if 'hard' in self.constrain_type:
                if d[1] is not None:
                    keys = []
                    for p in d[1]:
                        p_tokens = self.tokenizer(' '+' '.join(p), max_length=80, return_tensors='pt', add_special_tokens=False) # pre-space is insertted in dict for dict constrain
                        keys.append(p_tokens['input_ids'][0].tolist())
                    batch_keys = [keys]
                else:
                    batch_keys = None
                real_max_unconstraint_len = max_unconstraint_len
            else:
                #batch_keys = [d[1]]
                #batch_labels = [d[2]]
                batch_keys = [[Key(ph,lab,group=gr,max_grp_uncs_len=max_grp_uncs_len) for ph,lab,gr in zip(d[1],d[2],d[6])]]
                real_max_unconstraint_len = max_unconstraint_len + d[3]*unkey_uc_ratio
                #print('real uncs len: %d=%d+%d*%f'%(real_max_unconstraint_len, max_unconstraint_len, d[3],unkey_uc_ratio))

            if 'dict' in self.constrain_type:
                not_translate_labels = policy.get('not_translate_labels',[])
                self.__add_dict(d[1], d[2], not_translate_labels) #in case no entry in dict

            if self.model.config.is_encoder_decoder:
                encoder_outputs = self.get_hidden_representation(
                    input_lines_src
                )
                input_ids = torch.full(
                    (batch_size, 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=next(self.model.parameters()).device,
                )
                attention_mask = input_lines_src['attention_mask']
            else:
                if False:
                    input_ids = torch.full(
                        (batch_size, 1),
                        self.tokenizer.bos_token_id,
                        dtype=torch.long,
                        device=next(self.model.parameters()).device,
                    )
                    input_ids_len = 1
                    attention_mask = torch.full(
                        (batch_size, 1),
                        1,
                        dtype=torch.long,
                        device=next(self.model.parameters()).device,
                    )
                else:
                    input_ids = input_lines_src['input_ids']
                    input_ids_len = input_ids.shape[-1]
                    input_ids = input_ids.unsqueeze(1).expand(batch_size, 1, input_ids_len)

                input_ids = input_ids.contiguous().view(
                    1, input_ids_len
                )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)


            pre_hyp_index = torch.tensor([i for i in range(batch_size)]).view(batch_size, 1)
            
            if self.model.config.is_encoder_decoder:
                state_init = (input_ids, encoder_outputs.last_hidden_state, attention_mask, pre_hyp_index)
            else:
                state_init = (input_ids, pre_hyp_index)

            self.batch_decode_count = 0
            self.sub_batch_decode_count = 0
            self.used_gpu_memory_max = 0
            allHyp, allKidx, allScores, all_tot_hyp_cnt, all_hyp_cache_cnt = self.grid_beam.beam_run(state_init, batch_keys,
                n_best = min(n_best, self.beam_size),
                max_unconstraint_len = real_max_unconstraint_len,
                min_len = min_len,
                policy = policy,
                resource = self.resource,
                repetition_penalty=repetition_penalty,
                bad_words_ids=bad_words_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)

            if self.batch_decode_count>0 or self.sub_batch_decode_count>0:
                print('total hyps %d, decoded hyps %d, cache hit rate %f max_used_gpu_mem %d batch_dec %d sub_batch_dec %d'%(all_tot_hyp_cnt,
                    all_hyp_cache_cnt, 1.0-(all_hyp_cache_cnt/all_tot_hyp_cnt), self.used_gpu_memory_max,
                    self.batch_decode_count, self.sub_batch_decode_count))
            else:
                if all_tot_hyp_cnt==0:
                    print('Error ============================ all_tot_hyp_cnt is 0')
                else:
                    print('total hyps %d, decoded hyps %d, cache hit rate %f'%(all_tot_hyp_cnt,
                        all_hyp_cache_cnt, 1.0-(all_hyp_cache_cnt/all_tot_hyp_cnt)))

            if decode_token:
                #print(allHyp)
                #print(allKidx)
                #print(allScores)
                tot_hyp_path =  len(allHyp)
                if tot_hyp_path > 0:
                    for ih in range(tot_hyp_path):
                        idxs = allKidx[ih]
                        score = allScores[0][ih]
                        hyp_inds = [int(x.cpu().numpy()) for x in allHyp[ih]]
                        if with_label:
                            hyp_indx_mix = []
                            for j,(tok,idx) in enumerate(zip(hyp_inds, idxs)):
                                if j==0 or idxs[j-1]!=idxs[j]:
                                    #print(d[2][idx][0])
                                    if idx!=-1:
                                        hyp_indx_mix.append(label_dict[d[2][idx][0]])
                                    else:
                                        hyp_indx_mix.append(label_dict[input_data.un_labeled])
                                hyp_indx_mix.append(tok)
                        else:
                            hyp_indx_mix = hyp_inds
                        pred = self.tokenizer.decode(hyp_indx_mix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        if output_fn:
                            output_fn(pred, idx=ih, score=score)
                        else:
                            ret_preds += [pred]
                else:
                    if output_fn:
                        output_fn(None, data=d, idx=ih)

        return ret_hyp, ret_preds


