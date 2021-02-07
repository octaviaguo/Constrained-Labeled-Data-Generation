"""Grid Beam Search"""
from typing import Iterable, List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F

class BeamState(object):

    def __init__(self, size, maxLen, vocab_size, bos, eos, init_states, constrain_type,
        keys, labels, do_sample, temperature, top_k, top_p, print_log):
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
       
        self.print_log = print_log
        self.size = size
        self.done = False
        self.bos = bos
        self.eos = eos
        self.vocab_size = vocab_size
        self.init_states = init_states
        if keys is not None:
            self.numC = sum([len(k) for k in keys])
            self.keys = keys
            self.labels = labels
        else:
            self.numC = 0
            self.keys = []
            self.labels = []
        self.constrain_type = constrain_type


        keys = [k for k in self.keys]
        labels=[k for k in self.labels]
        close_w = []
        self.end_t = 0
        self.Grid = [[[] for i in range(self.numC+1)] for j in range(maxLen)]
        #self.Grid[0][0].append({'isOpen':True, 'y':self.tt.LongTensor(1).fill_(self.bos)[0], 'state': init_states,
        #        'score':self.tt.FloatTensor(1).zero_()[0],
        self.Grid[0][0].append({'isOpen':True, 'y':torch.LongTensor(1).fill_(self.bos)[0], 'state': init_states,
                'score':torch.FloatTensor(1).zero_()[0],
                'keys':keys, 'labels':labels, 'close_w':close_w})

        self.Cand = [[] for i in range(self.numC+1)]


    def update(self, c, workd_lk):
        #num_words = workd_lk.size(1) #workd_lk size [4,30004] ([beam_size, vob_size])
        beam_lk_list = []
        ks = []
        cand_cnt = 0
        new_hyp = []
        for i,cand in enumerate(self.Cand[c]):
            if cand['type'] == 'range':
                beam_lk_list.append(workd_lk[i] + cand['score'].expand_as(workd_lk[i]))
                y_range = cand['y']
                cand_start = cand_cnt
                cand_cnt += y_range[1]-y_range[0]
                ks.append({'start':cand_start, 'end':cand_cnt, 'hyp_idx':cand['hyp_idx']})
            elif cand['type'] == 'list':
                for y in cand['y']:
                    beam_lk_list.append(workd_lk[i][y] + cand['score'].unsqueeze(0))
                y_list = cand['y']
                cand_start = cand_cnt
                cand_cnt += len(y_list)
                ks.append({'start':cand_start, 'end':cand_cnt, 'hyp_idx':cand['hyp_idx']})
        
        beam_lk = torch.cat(beam_lk_list)

        _scores = beam_lk.view(-1)
        #bestScores shape [4], bestScoresId shape [4], 4 is beam_size
        k = min(self.size, _scores.size(0))

        if self.do_sample:
            if self.temperature != 1.0:
                _scores = _scores / self.temperature
            # Top-p/top-k filtering
            _scores = top_k_top_p_filtering(
                _scores, top_k=self.top_k, top_p=self.top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together to sample from all beam_idxs
            #_scores = _scores.contiguous().view(
            #    batch_size, num_beams * vocab_size
            #)  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)
            if self.print_log:
                print(probs)
            next_scores_id = torch.multinomial(probs, num_samples=k)  # (batch_size, num_beams * 2)
            # Compute next scores
            next_scores = torch.gather(_scores, -1, next_scores_id)  # (batch_size, num_beams * 2)
            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores, next_scores_indices = torch.sort(next_scores, descending=True)
            next_scores_id = torch.gather(next_scores_id, -1, next_scores_indices)  # (batch_size, num_beams * 2)
        else:
            next_scores, next_scores_id = _scores.topk(k, 0, True, True)
            #bestScores, bestScoresId
        #self.scores = next_scores
        for sc,idx in zip(next_scores,next_scores_id):
            for jj in range(len(ks)):
                if ks[jj]['end']>idx:
                    break
            cand = self.Cand[c][jj]
            ridx = idx-ks[jj]['start']
            if cand['type']=='range':
                y = cand['y'][0]+ridx
                keys = cand['keys']
                labels = cand['labels']
                close_w = cand['close_w']
                is_open = True
                #print(cand['y'], ridx, y)
            elif cand['type']=='list':
                y = torch.tensor(cand['y'][ridx])
                keys = cand['keys_list'][ridx]
                labels = cand['label_list'][ridx]
                close_w = cand['close_w_list'][ridx]
                is_open = len(close_w)==0
                #print(cand['y'], ridx, y)
            new_hyp.append({'isOpen':is_open, 'y':y, 'score': sc, 'pre_hyp_idx':ks[jj]['hyp_idx'],
                'keys':keys, 'labels':labels, 'close_w':close_w,
                'batch_proc_idx':jj})

        # End condition is when top-of-beam is EOS.
        if new_hyp[0]['y'] == self.eos and c==self.numC:
            self.done = True

        return new_hyp

    def sort_best(self):
        """Sort the beam."""
        scores = torch.stack([h['score'] for h in self.Grid[self.end_t][self.numC]])
        return torch.sort(scores, 0, True)


    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        grid_hyps = self.Grid[self.end_t][self.numC]
        label_index = {} #key: time t; val: label
        finded_key = 0

        for t in range(self.end_t-1, -1, -1):
            added_y = grid_hyps[k]['y']
            hyp.append(added_y) # have added keys
            pre_idx = grid_hyps[k]['pre_hyp_idx']
            #print(grid_hyps[k]['y'], self.id2word[grid_hyps[k]['y'].item()], pre_idx)
            grid_hyps = self.Grid[t][pre_idx[0]]
            k = pre_idx[1]

            if len(grid_hyps[k]['keys'])==(finded_key+1):
                def myfind(keys_digit, tgt):
                    for ii, i in enumerate(keys_digit):
                        if i[0]==tgt:
                            return ii
                    assert False

                finded_key += 1     
                key_idx = myfind(grid_hyps[k]['keys'], added_y)    
                label_index[str(t)]=grid_hyps[k]['labels'][key_idx] # label occurs at time t-1

        return hyp[::-1], label_index


class GridBeamSearch(object):

    def __init__(self, beam_size, decode_fn, vocab_size, bos, eos, constrain_type='', print_log=False):
        """
        tgt_tokens: {'bos',xx, 'eos',xx}
        constrain_type: ['','seq']
        """

        self.decode_fn = decode_fn
        self.bos = bos
        self.eos = eos
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.constrain_type = constrain_type
        self.print_log = print_log

    def beam_run(self, batch_init_states, batch_keys=None, batch_labels=None,
        max_length = 20,
        repetition_penalty=1.0,
        bad_words_ids=None,
        no_repeat_ngram_size=0,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0
        ):
        maxLen = max_length
        batch_size = len(batch_init_states[0])
        if batch_keys is None:
            batch_keys = [None]*batch_size
        if batch_labels is None:
            batch_labels = [None]*batch_size
        beam_size = self.beam_size

        #print('batch_size=',batch_size)
        beam = [
            BeamState(beam_size, maxLen+1, self.vocab_size, self.bos, self.eos, [ss[k] for ss in batch_init_states],
                constrain_type=self.constrain_type, keys=batch_keys[k],labels=batch_labels[k],
                do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                print_log=self.print_log)
                for k in range(batch_size)
        ]
        active_beam = [b for b in beam]

        for t in range(1, maxLen):
            if self.print_log:
                print('t=',t)
            hyp_y_set = []
            hyp_states_set = [[] for i in range(len(batch_init_states))]
            tot_hyp_cnt = 0
            remaining_sents = len(active_beam)
            batch_start = []
            for bi,b in enumerate(active_beam):
                batch_start.append(tot_hyp_cnt)
                for c in range(max(0, (b.numC+t)-maxLen), min(t, b.numC)+1):
                    if self.print_log:
                        print('c=',c)
                    b.Cand[c] = []
                    for jj,hyp in enumerate(b.Grid[t-1][c]):
                        if hyp['isOpen']:
                            if self.print_log:
                                print('generate %d'%tot_hyp_cnt)
                            hyp_y_set.append(hyp['y'])
                            for ii,s in enumerate(hyp['state']):
                                hyp_states_set[ii].append(s)
                            b.Cand[c].append({'type':'range', 'y':[0,b.vocab_size], 'score':hyp['score'], 'hyp_idx':[c,jj],
                                'keys':hyp['keys'], 'labels':hyp['labels'], 'close_w':hyp['close_w']
                                })
                            tot_hyp_cnt += 1
                    if c > 0:
                        for jj,hyp in enumerate(b.Grid[t-1][c-1]):
                            hyp_y_set.append(hyp['y'])
                            for ii,s in enumerate(hyp['state']):
                                hyp_states_set[ii].append(s)
                            
                            if hyp['isOpen']:
                                if self.print_log:
                                    print('open %d'%tot_hyp_cnt, hyp['keys'])
                                y_list = []
                                close_w_list = []
                                keys_list = []
                                label_list= []
                                for ii,key in enumerate(hyp['keys']):
                                    y_list.append(key[0])
                                    keys_list.append([k for i,k in enumerate(hyp['keys']) if i!=ii])
                                    label_list.append([k for i,k in enumerate(hyp['labels']) if i!=ii])
                                    close_w_list.append([w for i,w in enumerate(key) if i>0])
                                    if b.constrain_type=='seq':
                                        break
                            else:
                                if self.print_log:
                                    print('close %d'%tot_hyp_cnt)
                                y_list = [hyp['close_w'][0]]
                                keys_list = [hyp['keys']]
                                label_list = [hyp['labels']]
                                close_w_list = [[w for i,w in enumerate(hyp['close_w']) if i>0]]

                            b.Cand[c].append({'type':'list', 'y':y_list, 'score':hyp['score'], 'hyp_idx':[c-1,jj],
                                'keys_list':keys_list, 'label_list':label_list, 'close_w_list':close_w_list
                                })
                            tot_hyp_cnt += 1
                    if self.print_log:
                        print('t=%d, b=%d, Cand[%d] cnt %d:'%(t,bi,c,tot_hyp_cnt), b.Cand[c])

            input = torch.stack([y.cpu() for y in hyp_y_set]).t().contiguous().view(-1,1)

            states = [torch.stack(s) for s in hyp_states_set]
            if self.print_log:
                print('decode batch size is ', input.shape[0])
            logits, states = self.decode_fn(input, states, t)
            
            next_token_logits = logits[:,-1,:]
            
            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=states[0], #input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=t,
                min_length=0,
                max_length=maxLen,
                eos_token_id=self.eos,
                repetition_penalty=repetition_penalty,
            )

            done_beam = []
            for idx,b in enumerate(active_beam):
                cand_start = 0
                for c in range(max(0, (b.numC+t)-maxLen), min(t, b.numC)+1):
                    if self.print_log:
                        print(b.Cand[c])
                    new_hyps = b.update(c, scores.data[batch_start[idx]+cand_start:batch_start[idx]+cand_start+len(b.Cand[c])].cpu())
                    cand_start += len(b.Cand[c])
                    for hyp in new_hyps:
                        b_idx = hyp['batch_proc_idx']
                        cur_states = []
                        for i,s in enumerate(states):
                            if s is None:
                                cur_states.append(b.Grid[0][0][0]['state'][i])
                            else:
                                #shape = s.shape[1:]
                                #cur_states.append(s.view(remaining_sents, tot_hyp_cnt, *shape)[idx,b_idx,:])
                                cur_states.append(s[batch_start[idx]+b_idx,:])
                        hyp['state'] = cur_states 

                    b.Grid[t][c] = new_hyps
                    b.end_t = t
                        

                if b.done:
                    done_beam.append(b)

            for b in done_beam:
                active_beam.remove(b)
            if len(active_beam)==0:
                break


        allHyp, allScores, allLab = [], [], []
        n_best = 1

        for b in range(batch_size):
            print('b=',b)
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            tmp_hyp_2 = []
            for k in ks[:n_best]:
                tmp_hyp, tmp_lab = beam[b].get_hyp(k)
                tmp_hyp_2.append(tmp_hyp)
                
            hyps = zip(*tmp_hyp_2)
            #hyps = zip(*[ beam[b].get_hyp(k) for k in ks[:n_best] ])
            allHyp += [hyps]
            allLab += [tmp_lab]
        return allHyp, allLab

    '''
    below code are based on generation_utils.py of huggingface
    '''
    def enforce_repetition_penalty_(self, lprobs, prev_output_tokens, repetition_penalty):
        """
        Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
        """
        for i in range(lprobs.shape[0]):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = scores.shape[0]
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # Exclude EOS token (already processed)
            bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
            # Modify the scores in place by setting the banned tokens logits to `-inf`
            set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

        return scores

def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

