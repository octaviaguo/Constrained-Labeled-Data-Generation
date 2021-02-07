"""Grid Beam Search"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from new_grid_beam import GridBeamSearch
from transformers.modeling_outputs import BaseModelOutput

class GridBeamSearchDecoder(object):
    """Beam Search decoder."""

    def __init__(
        self,
        model,
        tokenizer,
        beam_size=1,
        constrain_type='',
        process_size=32,
        print_log=False,
        carry_label=False
    ):
        self.beam_size = beam_size
        self.constrain_type=constrain_type
        self.model = model
        self.tokenizer = tokenizer
        self.past = None
        self.carry_label=carry_label

        def decode(input, states, t):
            input_ids = states[0]
            last_hidden_state = states[1]
            attention_mask = states[2]
            pre_hyp_index = states[3]
            if t>1:
                input_ids = torch.cat([input_ids, input.to(self.model.device)], dim=-1)
                past = self.model._reorder_cache(self.past, pre_hyp_index.squeeze(-1).to(self.model.device))
            else:
                past = None
            encoder_outputs = BaseModelOutput(
                last_hidden_state=states[1],
                hidden_states=None,
                attentions=None
            )

            model_inputs = {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": input_ids.to(self.model.device),
                "attention_mask": attention_mask.to(self.model.device),
                "use_cache": True,  # change this to avoid caching (presumably for debugging)
            }

            outputs = self.model(**model_inputs, return_dict=True)

            self.past = outputs.past_key_values
            cur_hyp_index = torch.tensor([i for i in range(input_ids.shape[0])]).view(input_ids.shape[0], 1)
            return outputs.logits, (input_ids, outputs.encoder_last_hidden_state, states[2], cur_hyp_index)

        '''
        sub batch decoding to avoid GPU running out of memory 
        '''
        def _get_buffer_cpu(attn_cache):
            #attn_cache_new = {}
            for k, input_buffer_k in attn_cache.items():
                if input_buffer_k is not None:
                    attn_cache[k] = input_buffer_k.cpu()
                    del input_buffer_k
            return attn_cache

        def _get_cache_cpu(past):
            reordered_past = []
            for layer_past in past:
                # get the correct batch idx from decoder layer's batch dim for cross and self-attn
                layer_past_new = {
                    attn_key: _get_buffer_cpu(attn_cache) for attn_key, attn_cache in layer_past.items()
                }
                reordered_past.append(layer_past_new)
            return reordered_past

        def _get_buffer(attn_cache, indice):
            attn_cache_new = {}
            for k, input_buffer_k in attn_cache.items():
                if input_buffer_k is not None:
                    attn_cache_new[k] = input_buffer_k.index_select(0, indice).to(self.model.device)
            return attn_cache_new

        def _get_cache(past, indice):
            reordered_past = []
            for layer_past in past:
                # get the correct batch idx from decoder layer's batch dim for cross and self-attn
                layer_past_new = {
                    attn_key: _get_buffer(attn_cache, indice) for attn_key, attn_cache in layer_past.items()
                }
                reordered_past.append(layer_past_new)
            return reordered_past

        def _concat_buffer(attn_cache, attn_cache_list):
            attn_cache_new = {}
            for k, input_buffer_k in attn_cache.items():
                if input_buffer_k is not None:
                    buffer_list = [attn_cache_list[i][k].cpu() for i in range(len(attn_cache_list))]
                    attn_cache_new[k] = torch.cat(buffer_list)
            return attn_cache_new

        def _concat_cache(past_list_):
            concated_past = []
            for l,layer_past in enumerate(past_list_[0]):
                # get the correct batch idx from decoder layer's batch dim for cross and self-attn
                layer_past_new = {
                    attn_key: _concat_buffer(attn_cache, [past_list_[i][l][attn_key] for i in range(len(past_list_))]) for attn_key, attn_cache in layer_past.items()
                }
                concated_past.append(layer_past_new)
            return concated_past

        def sub_batch_decode(input, states, t):
            with torch.no_grad():
                use_cache = True #False only for debugging
                input_ids = states[0]
                last_hidden_state = states[1]
                attention_mask = states[2]
                pre_hyp_index = states[3]
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
                for i in range(0, states[0].shape[0], process_size):
                    if print_log:
                        print('=============before decoding')
                        print(torch.cuda.memory_reserved())

                    if t>1 and use_cache:
                        past_index = torch.tensor([i+j for j in range(process_size) if (i+j)<states[0].shape[0]])
                        past_i = _get_cache(past, past_index)
                    else:
                        past_i = None
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
                        "use_cache": True,  # change this to avoid caching (presumably for debugging)
                    }
                    outputs = self.model(**model_inputs, return_dict=True)
                    #print(outputs)
                    logits_list.append(outputs.logits.cpu())
                    hidden_state_list.append(outputs.encoder_last_hidden_state.cpu())

                    del outputs.logits
                    del outputs.encoder_last_hidden_state
                    del last_hidden_state_
                    del input_ids_
                    del attention_mask_
                    if use_cache:
                        past_list.append(_get_cache_cpu(outputs.past_key_values))
                        if past_i is not None:
                            _get_cache_cpu(past_i) # call this function to delete cuda tensor
                    torch.cuda.empty_cache()
                    if print_log:
                        print('===============after decoding')
                        print(torch.cuda.memory_reserved())
                logits = torch.cat(logits_list)
                encoder_last_hidden_state = torch.cat(hidden_state_list)
                if use_cache:
                    self.past = _concat_cache(past_list)
                cur_hyp_index = torch.tensor([i for i in range(input_ids.shape[0])]).view(input_ids.shape[0], 1)
            return logits, (input_ids.cpu(), encoder_last_hidden_state, states[2], cur_hyp_index)


        self.grid_beam = GridBeamSearch(
            beam_size=self.beam_size,
            decode_fn=decode if process_size<0 else sub_batch_decode,
            bos = self.model.config.decoder_start_token_id,
            eos = self.tokenizer.eos_token_id,
            vocab_size=self.model.config.vocab_size, 
            constrain_type=self.constrain_type,
            print_log=print_log)


    def get_hidden_representation(self, input_ids):
        """Get hidden representation for a sentence."""
        #attention_mask = input_ids.new_ones(input_ids.shape)
        encoder = self.model.get_encoder()
        encoder_output = encoder(input_ids['input_ids'].to(self.model.device), attention_mask=input_ids['attention_mask'].to(self.model.device), return_dict=True)

        return encoder_output


    def decode(self, batch_sent, batch_key_phrases, batch_label_phrases=None,
        max_length=20, 
        decode_token=True,
        repetition_penalty=1.0,
        bad_words_ids=None,
        no_repeat_ngram_size=0,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0
        ):
        input_lines_src = self.tokenizer(batch_sent, max_length=80, return_tensors='pt', padding=True, truncation=True)

        batch_keys = []
        batch_labels = []
        if batch_key_phrases is not None:
            for key_phrases in batch_key_phrases:
                keys = []
                key_phrases = key_phrases.split('<mask>')
                #key_tokens = self.tokenizer(key_phrases, max_length=max_length, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
                #batch_keys.append(key_tokens['input_ids'])
                for p in key_phrases:
                    if len(p.strip())==0:
                        continue
                    p_tokens = self.tokenizer(p, max_length=80, return_tensors='pt', add_special_tokens=False)
                    keys.append(p_tokens['input_ids'][0].tolist())
                batch_keys.append(keys)
                
            if batch_label_phrases is not None:
                for label_phrases in batch_label_phrases:
                    labels = []
                    label_phrases = label_phrases.split('<mask>')
                    for p in label_phrases:
                        if len(p.strip())==0:
                            continue
                        labels.append(p) #???
                    batch_labels.append(labels)
            else:
                batch_labels = [['' for k in keys] for keys in batch_keys]
        else:
            batch_keys = None
            batch_labels = None

        print('batch_keys:\t', batch_keys)

        encoder_outputs = self.get_hidden_representation(
            input_lines_src
        )
        batch_size = len(batch_sent)
        input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        pre_hyp_index = torch.tensor([i for i in range(batch_size)]).view(batch_size, 1)
        
        state_init = (input_ids, encoder_outputs.last_hidden_state, input_lines_src['attention_mask'], pre_hyp_index)
        allHyp, allLab = self.grid_beam.beam_run(state_init, batch_keys, batch_labels,
            max_length = max_length,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)

        if decode_token:
            all_hyp_inds = [[int(x[0].cpu().numpy()) for x in hyp] for hyp in allHyp]
            all_preds = [self.tokenizer.decode(hyp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for hyp in all_hyp_inds]
            # add labels !!!
            for prediction in all_preds[0]:
                todo = prediction.split()
                for i in range(len(todo)):
                    if str(i) in allLab:
                        todo[i] = todo[i] + '<label>'+ allLab[str(i)]

        else:
            all_preds = None

        return allHyp, all_preds



