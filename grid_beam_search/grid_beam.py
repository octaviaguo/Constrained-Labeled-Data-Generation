"""Grid Beam Search"""
from typing import Iterable, List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F
from threading import Thread

class Key(object):
    def __init__(self, ph, lab='', group=-1, max_grp_uncs_len=0):
        self.ph = ph
        self.lab = lab
        self.group = group
        self.idx = 0
        self.insert_vocab = None
        self.max_grp_uncs_len = max_grp_uncs_len


class HardConstraint(object):
    def __init__(self, type, keys=None, policy={}, resource={}, debug_flag=False):
        self.typ = type
        self.policy=policy
        self.resource=resource
        self.debug_flag = debug_flag
        self.done = False
        self.isOpen = True        
        self.keys = keys
        if keys is not None:
            self.keys_idx = [i for i in range(len(keys))]
        self.kidx = -1
        self.close_w = []
        if keys is None or len(keys)==0:
            self.done = True

    @classmethod
    def __clone_constraint(self, constraint_ins, copy_list=False):
        ret_constraint = HardConstraint(constraint_ins.typ, policy=constraint_ins.policy, resource=constraint_ins.resource)
        ret_constraint.isOpen = constraint_ins.isOpen
        ret_constraint.done = constraint_ins.done
        ret_constraint.kidx = constraint_ins.kidx
        ret_constraint.debug_flag = constraint_ins.debug_flag
        if copy_list:
            ret_constraint.keys = [keys for keys in constraint_ins.keys]
            ret_constraint.keys_idx = [k for k in constraint_ins.keys_idx]
            ret_constraint.close_w = [w for w in constraint_ins.close_w]
        return ret_constraint

    def __update_state(self):
        self.isOpen = len(self.close_w)==0
        self.done = self.isOpen and len(self.keys)==0

    def generate_cands(self):
        y_list = []
        constraint_list = []
        kidx_list = []
        if self.isOpen:
            for ii,key in enumerate(self.keys):
                new_constraint = HardConstraint.__clone_constraint(self)
                y_list.append(key[0])
                new_constraint.keys = [k for i,k in enumerate(self.keys) if i!=ii]
                new_constraint.keys_idx = [k for i,k in enumerate(self.keys_idx) if i!=ii]
                new_constraint.close_w = [w for i,w in enumerate(key) if i>0]
                kidx_list.append(self.keys_idx[ii])
                new_constraint.kidx = self.keys_idx[ii]
                new_constraint.__update_state()
                constraint_list.append(new_constraint)
                if self.policy.get('keep_order',False):
                    break
        else:
            new_constraint = HardConstraint.__clone_constraint(self)
            y_list = [self.close_w[0]]
            kidx_list = [self.kidx]
            new_constraint.keys = self.keys
            new_constraint.keys_idx = self.keys_idx
            new_constraint.close_w = [w for i,w in enumerate(self.close_w) if i>0]
            new_constraint.__update_state()
            constraint_list.append(new_constraint)
        return y_list, constraint_list, kidx_list

    def dump(self, logger):
        if self.isOpen:
            log_txt = 'remains keys: '+ ','.join(['['+','.join([w for w in key])+'('+str(i)+')]' for key,i in zip(self.keys,self.keys_idx)])
            logger.add(log_txt)

class DictConstraint(object):
    def __init__(self, type, keys=None, policy={}, resource={}, debug_flag=False):
        self.typ = type
        self.policy = policy
        self.resource = resource
        self.done = False
        self.isOpen = True  
        self.inGroup = False      
        self.src_keys = keys
        self.key = None #cur key
        self.group = -1
        self.grp_uncs_len = 0
        self.debug_flag = debug_flag
        if keys is not None:
            for i,key in enumerate(keys):
                key.idx = i
        self.kidx = -1
        self.src_close_w = []
        if keys is None or len(keys)==0:
            self.done = True
        self.trg_close_tokens = None
        self.o_top_th = policy.get('o_top_th',0.0)
        self.ne_top_th = policy.get('ne_top_th',0.0)
        self.dict_policy = policy.get('dict_policy', '')
        self.keep_order = policy.get('keep_order',False)
        self.ph_keep_order = policy.get('ph_keep_order',False)
        self.gp_keep_order = policy.get('gp_keep_order',False)
        self.not_translate_labels = policy.get('not_translate_labels',[])
        self.dict = resource['dict']
        self.trg_token_dict = resource.get('trg_token_dict',{})

        ##for debug_flag == True
        self.id = '0'
        self.clone_count = 0
        self.info = {}
        self.cand_word = 0

    @classmethod
    def __clone_constraint(self, constraint_ins, copy_list=False):
        ret_constraint = DictConstraint(constraint_ins.typ, policy=constraint_ins.policy, resource=constraint_ins.resource)
        ret_constraint.cand_word = constraint_ins.cand_word
        ret_constraint.isOpen = constraint_ins.isOpen
        ret_constraint.done = constraint_ins.done
        ret_constraint.kidx = constraint_ins.kidx
        #ret_constraint.lab = constraint_ins.lab
        ret_constraint.key = constraint_ins.key
        ret_constraint.group = constraint_ins.group
        ret_constraint.grp_uncs_len = constraint_ins.grp_uncs_len
        ret_constraint.debug_flag = constraint_ins.debug_flag
        if copy_list:
            ret_constraint.src_keys = [keys for keys in constraint_ins.src_keys]
            ret_constraint.src_close_w = [w for w in constraint_ins.src_close_w]
            ret_constraint.trg_close_tokens = [t for t in constraint_ins.trg_close_tokens]
        if constraint_ins.debug_flag:
            ret_constraint.id = constraint_ins.id + '-' + str(constraint_ins.clone_count)
            constraint_ins.clone_count+=1
        return ret_constraint

    def __update_state(self):
        self.isOpen = len(self.trg_close_tokens)==0 and len(self.src_close_w)==0
        self.done = self.isOpen and len(self.src_keys)==0
        if self.group<=0:
            self.inGroup = False
        else:
            remain_groups = [key.group for key in self.src_keys]
            self.inGroup = self.group in remain_groups
        #if self.inGroup:
        #    self.insert_vocab = 

    def is_open(self):
        ret = False
        if self.inGroup:
            if self.grp_uncs_len<self.key.max_grp_uncs_len:
                ret = self.isOpen
        else:
            ret = self.isOpen
        return ret

    def get_token_keys_list(self, w, lab):
        if lab in self.not_translate_labels:
            token_keys_list = [self.trg_token_dict[w]]
            #print('PPPP:', w, token_keys_list)
        else:
            trg_w_items = self.dict.get(w,[])
            if (lab=='[O]' and 'O_best' in self.dict_policy) or (lab!='[O]' and 'NE_best' in self.dict_policy):
                token_keys_list = [trg_w_items[0][2]]
            else:
                #token_keys_list = [w_itm[2] for w_itm in trg_w_items]
                if lab=='[O]':
                    th = self.o_top_th
                else:
                    th = self.ne_top_th
                max_p = max([w_itm[1] for w_itm in trg_w_items])
                token_keys_list = [w_itm[2] for w_itm in trg_w_items if w_itm[1]>=(max_p*th)]
        return token_keys_list

    def generate_cands(self):
        y_list = []
        constraint_list = []
        kidx_list = []
        count = 0
        if self.isOpen:
            remain_groups = [key.group for key in self.src_keys]
            cand_groups = []
            for ii, key in enumerate(self.src_keys):
                if self.keep_order and ii>0:
                    break
                if self.group!=key.group and self.group in remain_groups:
                    continue
                if self.gp_keep_order:
                    if key.group in cand_groups:
                        break
                    cand_groups.append(key.group)
                new_src_keys = [k for i,k in enumerate(self.src_keys) if i!=ii]
                for jj,w in enumerate(key.ph):
                    if (self.ph_keep_order or key.lab[0] in self.not_translate_labels) and jj>0:
                        break
                    new_src_close_w = [w for i,w in enumerate(key.ph) if i!=jj]
                    token_keys_list = self.get_token_keys_list(w, key.lab[0])
                    if len(token_keys_list)==0:
                        print('Warning, no entry for %s in dict'%w)
                    for o, token_keys in enumerate(token_keys_list):
                        y_list.append(token_keys[0])
                        trg_close_tokens = token_keys[1:]
                        new_constraint = DictConstraint.__clone_constraint(self)
                        new_constraint.src_keys = new_src_keys
                        new_constraint.src_close_w = new_src_close_w
                        new_constraint.trg_close_tokens = trg_close_tokens
                        #new_constraint.lab = key.lab[0]
                        new_constraint.key = key
                        new_constraint.group = key.group
                        if self.group<=0 or self.group!=key.group:
                            new_constraint.grp_uncs_len = 0
                        kidx_list.append(self.src_keys[ii].idx)
                        new_constraint.kidx = self.src_keys[ii].idx
                        new_constraint.__update_state()
                        if self.debug_flag:
                            new_constraint.cand_word = '-'.join([str(t) for t in token_keys])
                            new_constraint.info = {'tran_opt':o, 'w':w}
                        constraint_list.append(new_constraint)
        else:
            if len(self.trg_close_tokens)>0:
                y_list = [self.trg_close_tokens[0]]
                kidx_list = [self.kidx]
                trg_close_tokens = [t for i,t in enumerate(self.trg_close_tokens) if i>0]

                new_constraint = DictConstraint.__clone_constraint(self)
                new_constraint.src_keys = self.src_keys
                new_constraint.src_close_w = self.src_close_w
                new_constraint.trg_close_tokens = trg_close_tokens
                new_constraint.__update_state()
                constraint_list.append(new_constraint)
            elif len(self.src_close_w)>0:
                for jj,w in enumerate(self.src_close_w):
                    if (self.ph_keep_order or self.key.lab[0] in self.not_translate_labels) and jj>0:
                        break
                    new_src_close_w = [w for i,w in enumerate(self.src_close_w) if i!=jj]
                    token_keys_list = self.get_token_keys_list(w, self.key.lab[0])
                    if len(token_keys_list)==0:
                        print('Warning, no entry for %s in dict'%w)
                    for o,token_keys in enumerate(token_keys_list):
                        y_list.append(token_keys[0])
                        kidx_list.append(self.kidx)
                        trg_close_tokens = token_keys[1:]
                        new_constraint = DictConstraint.__clone_constraint(self)
                        new_constraint.src_keys = self.src_keys
                        new_constraint.src_close_w = new_src_close_w
                        new_constraint.trg_close_tokens = trg_close_tokens
                        new_constraint.__update_state()
                        if self.debug_flag:
                            new_constraint.cand_word = '-'.join([str(t) for t in token_keys])
                            new_constraint.info = {'tran_opt':o, 'w':w}
                        constraint_list.append(new_constraint)

        return y_list, constraint_list, kidx_list

    def dump(self, logger):
        if self.isOpen:
            log_txt = 'remains keys: '+ ','.join(['['+','.join([w for w in key.ph])+'('+str(key.idx)+')]' for key in self.src_keys])
            logger.add(log_txt)

    def get_dump_info(self, logger):
        if 'tran_opt' not in self.info.keys():
            log_txt = '<trgclose> '
        else:
            log_txt = ''
        log_txt += 'id:%s '%(self.id)
        log_txt += 'trg:%s'%self.cand_word
        return log_txt


class DictFilter(object):
    def __init__(self, type, policy, resource, debug_flag=False):
        self.typ = type
        self.debug_flag = debug_flag
        self.vocab = resource['filter_vocab']
        self.policy = policy
        self.resource = resource
        self.tokens_list = [[]]
        self.yfilter = []

    def generate_yfilter(self, vocab=None, max_len=16):
        if vocab is None:
            vocab = self.vocab
        new_tokens_list = []
        self.yfilter = []
        for tokens in self.tokens_list:
            if len(tokens)==0:
                new_tokens_list+=[t[1:] for t in vocab if len(t)<=max_len]
                self.yfilter+=[t[0] for t in vocab if len(t)<=max_len]
            else:
                new_tokens_list.append(tokens[1:])
                self.yfilter.append(tokens[0])
        self.tokens_list = new_tokens_list

    @classmethod
    def __clone_filter(self, filter_ins, copy_list=False):
        ret_filter = DictFilter(filter_ins.typ, policy=filter_ins.policy,
            resource=filter_ins.resource,
            debug_flag=filter_ins.debug_flag)
        return ret_filter

    def is_open(self):
        return ([] in self.tokens_list)

    def get_new_filter(self):
        return DictFilter.__clone_filter(self)

    def get_updated_filter(self, y):
        y = y.item()
        new_tokens_list = []
        done = 0
        for cand, tokens in zip(self.yfilter, self.tokens_list):
            if cand==y:
                if len(tokens)>0:
                    new_tokens_list.append(tokens)
                else:
                    done+=1
        if done==0 and len(new_tokens_list)==0:
            print('get_updated_filter error:', y)
            print(self.yfilter)
            #input('error')
        if done>0:
            new_tokens_list.append([])
        updated_filter = DictFilter.__clone_filter(self)
        updated_filter.tokens_list = new_tokens_list
        return updated_filter


class Hyp(object):
    def __init__(self, t, c, y, score, uc_path_len, cs, kidx=-1, pre_hyp_idx=-1, batch_proc_idx=-1, states=None):
        self.t = t
        self.c = c
        self.y = y
        self.kidx = kidx
        self.score = score
        self.pre_hyp_idx = pre_hyp_idx
        self.uc_path_len = uc_path_len
        self.cs = cs
        self.batch_proc_idx = batch_proc_idx
        self.states = states
        self.cf = None
        #for debug
        self.idx = 0
        self.cand_index = -1
        self.rank = []

    def dump(self, logger):
        log_txt = '\t\t%sHyp(%d,%d,%d): '%('*' if 0 in self.rank else '', self.t, self.c, self.idx)
        log_txt += 'score %f '%(self.score)
        log_txt += 'cand_index %d '%(self.cand_index[1])
        log_txt += 'rank['+','.join([str(r) for r in self.rank])+'] '
        logger.add(log_txt)

class Cand(object):
    def __init__(self, type, y, score, hyp_idx, uc_path_len, cs_list, kidx_list,cf):
        self.typ=type
        self.y=y
        self.score=score
        self.hyp_idx=hyp_idx
        self.uc_path_len=uc_path_len
        self.cs_list=cs_list
        self.kidx_list=kidx_list
        self.cf=cf
        self.cache_i = -1
        ##debug_flag==True
        self.pre_cs=None
        self.topk_local_score=None
        self.topk_cand_score=None
        self.local_score=None

    def __repr__(self):
        return f'Cand(type={self.typ},y={self.y},score={self.score},hyp_idx={self.hyp_idx},uc_path_len={self.uc_path_len},kidx_list={self.kidx_list},cache_i={self.cache_i})'

    def dump(self, t, c, hyp_dump_fn, logger):
        if self.pre_cs and self.pre_cs.isOpen:
            log_txt = 'Op'
        elif self.typ=='range':
            log_txt = 'Ra'
        else:
            log_txt = 'Cl'
        log_txt += 'Cand pre(%d,%d,%d)->(%d,%d):'%(t-1,self.hyp_idx[0], self.hyp_idx[1],t,c)

        log_txt +=  '%s score %f uc_path_len %d'%(self.typ, self.score, self.uc_path_len)
        logger.add(log_txt)
        if self.pre_cs is not None:
            self.pre_cs.dump(logger)
        if self.typ=='list':
            for i,y in enumerate(self.y):
                w = logger.tokenizer._convert_id_to_token(y)
                log_txt = '\t%d:keys[%d]:%d(%s), score %f %s '%(i,self.kidx_list[i],y,w,self.local_score[i],
                    '' if self.cs_list[i].isOpen else '<close> ')
                tran_opt = self.cs_list[i].info.get('tran_opt',-1)
                if tran_opt>=0:
                    src_w = self.cs_list[i].info['w']
                    trg_w=logger.get_tran_words(src_w)[tran_opt]
                    log_txt += 'tran[%s,%d]=%s(%f) '%(src_w, tran_opt, trg_w[0],trg_w[1])
                log_txt += self.cs_list[i].get_dump_info(logger)
                logger.add(log_txt)
                hyp_dump_fn(i)
        else:
            log_txt='\t%s'%('local score:'+' '.join([str(idx.item())+'('+logger.tokenizer._convert_id_to_token(idx.item())+')'+'%f'%sc.item() for sc,idx in zip(*self.topk_local_score)]))
            logger.add(log_txt)
            log_txt='\t%s'%('topk score:'+' '.join([str(idx.item())+'('+logger.tokenizer._convert_id_to_token(idx.item())+')'+'%f'%sc.item() for sc,idx in zip(*self.topk_cand_score)]))
            logger.add(log_txt)
            hyp_dump_fn(0)

class BeamState(object):
    maxC = 256 #128
    maxT = 256
    def __init__(self, size, max_unconstraint_len, vocab_size, bos, eos, init_states, constrain_type,
        keys, policy, resource,
        do_sample, temperature, top_k, top_p, print_log, debug_flag):
        self.policy = policy
        self.score_decay = policy.get('score_decay', 1.0)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
       
        self.print_log = print_log
        self.max_unconstraint_len = max_unconstraint_len
        #print('max_unconstraint_len ', max_unconstraint_len)
        self.size = size
        self.cs_beam_size = policy.get('cs_beam_size', 0)
        self.tot_hyp_cnt = 0
        self.done = False
        self.bos = bos
        self.eos = eos
        self.vocab_size = vocab_size
        self.y_set = set([y for y in range(vocab_size)])
        self.filter_vocab = resource.get('filter_vocab',None)
        self.init_states = init_states
        self.debug_flag = debug_flag
        if keys is not None:
            #self.numC = sum([len(k) for k in keys])
            self.keys = keys
        else:
            #self.numC = 0
            self.keys = []
        self.constrain_type = constrain_type


        keys = [k for k in self.keys]
        self.end_t = 0
        self.Grid = [[[] for i in range(BeamState.maxC+1)] for j in range(BeamState.maxT)]
        if self.debug_flag:
            self.GridCand = [[[] for i in range(BeamState.maxC+1)] for j in range(BeamState.maxT)]
        #self.Grid[0][0].append({'isOpen':True, 'y':self.tt.LongTensor(1).fill_(self.bos)[0], 'state': init_states,
        #        'score':self.tt.FloatTensor(1).zero_()[0],
        if 'hard' in constrain_type:
            constraint = HardConstraint('hard', keys=keys, policy=policy, resource=resource, debug_flag=self.debug_flag)
        else:
            constraint = DictConstraint('dict', keys=keys, policy=policy, resource=resource, debug_flag=self.debug_flag)

        self.Grid[0][0].append(
                Hyp(
                t=0, c=0,
                y=torch.LongTensor(1).fill_(self.bos)[0], 
                states=init_states,
                score=torch.FloatTensor(1).zero_()[0],
                uc_path_len=0,
                cs=constraint
                ))
        if self.filter_vocab is not None:
            cand_filter = DictFilter('', policy=policy, resource=resource, debug_flag=self.debug_flag)
            self.Grid[0][0][0].cf = cand_filter

        self.Cand = [[] for i in range(BeamState.maxC+1)]


    def update(self, t, c, workd_lk):
        #num_words = workd_lk.size(1) #workd_lk size [4,30004] ([beam_size, vob_size])
        beam_lk_list = []
        ks = []
        cand_cnt = 0
        new_hyps = []
        for i,cand in enumerate(self.Cand[c]):
            path_score = cand.score*self.score_decay
            #print(path_score)
            if cand.typ == 'range':
                if cand.cf is not None:
                    indices_to_remove = list(self.y_set.difference(cand.cf.yfilter))
                    #print(cand.cf.yfilter)
                    workd_lk[i,indices_to_remove] = -float("Inf")
                beam_lk_list.append(workd_lk[i] + path_score.expand_as(workd_lk[i]))
                y_range = cand.y
                cand_start = cand_cnt
                cand_cnt += y_range[1]-y_range[0]
                ks.append({'start':cand_start, 'end':cand_cnt, 'hyp_idx':cand.hyp_idx})
                if self.debug_flag:            
                    cand_scores, cand_scores_id = torch.cat([workd_lk[i] + path_score.expand_as(workd_lk[i])]).view(-1).topk(self.size, 0, True, True)
                    local_scores, local_scores_id = torch.cat([workd_lk[i]]).view(-1).topk(self.size, 0, True, True)
                    cand.topk_cand_score = (cand_scores, cand_scores_id)
                    cand.topk_local_score = (local_scores, local_scores_id)
            elif cand.typ == 'list':
                for y in cand.y:
                    beam_lk_list.append(workd_lk[i][y] + path_score.unsqueeze(0))
                y_list = cand.y
                if self.debug_flag:
                    cand.local_score=workd_lk[i][y_list]
                cand_start = cand_cnt
                cand_cnt += len(y_list)
                ks.append({'start':cand_start, 'end':cand_cnt, 'hyp_idx':cand.hyp_idx})

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
                print(probs, file=self.print_log)
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
            if sc==-float("Inf"):
                continue
            for jj in range(len(ks)):
                if ks[jj]['end']>idx:
                    break
            cand = self.Cand[c][jj]
            ridx = idx-ks[jj]['start']
            new_filter = None
            if cand.typ=='range':
                y = cand.y[0]+ridx
                constraint = cand.cs_list[0]
                kidx = cand.kidx_list[0]
                is_open = True
                if cand.cf is not None:
                    new_filter = cand.cf.get_updated_filter(y)
                #print(cand.y, ridx, y)
            elif cand.typ=='list':
                y = torch.tensor(cand.y[ridx])
                constraint = cand.cs_list[ridx]
                kidx = cand.kidx_list[ridx]
                if cand.cf is not None:
                    new_filter = cand.cf.get_new_filter()
                #print(cand.y, ridx, y)
            new_hyp = Hyp(
                t=t, c=c,
                y=y, kidx=kidx, score=sc, pre_hyp_idx=ks[jj]['hyp_idx'],
                uc_path_len=cand.uc_path_len,
                cs=constraint,
                batch_proc_idx=jj
                )
            new_hyp.cf = new_filter
            #for dump
            new_hyp.cand_index = [jj,ridx]
            new_hyp.idx = len(new_hyps)
            ##
            new_hyps.append(new_hyp)

        # End condition is when top-of-beam is EOS.
        if new_hyps[0].y == self.eos and new_hyps[0].cs.done:
            self.done = True
        return new_hyps


    def double_topk_update(self, t, c, workd_lk):
        #num_words = workd_lk.size(1) #workd_lk size [4,30004] ([beam_size, vob_size])
        #print('------------------------')
        beam_lk_list = []
        cs_beam_lk_list = []

        ks = []
        cand_cnt = 0
        new_hyps = []
        for i,cand in enumerate(self.Cand[c]):
            path_score = cand.score*self.score_decay
            #print(path_score)
            if cand.typ == 'range':
                if cand.cf is not None:
                    indices_to_remove = list(self.y_set.difference(cand.cf.yfilter))
                    #print('yfilter:', cand.cf.yfilter)
                    workd_lk[i,indices_to_remove] = -float("Inf")
                beam_lk_list.append(workd_lk[i] + path_score.expand_as(workd_lk[i]))
                y_range = cand.y
                cand_start = cand_cnt
                cand_cnt += y_range[1]-y_range[0]
                ks.append({'start':cand_start, 'end':cand_cnt, 'hyp_idx':cand.hyp_idx})
                if self.debug_flag:            
                    cand_scores, cand_scores_id = torch.cat([workd_lk[i] + path_score.expand_as(workd_lk[i])]).view(-1).topk(self.size, 0, True, True)
                    local_scores, local_scores_id = torch.cat([workd_lk[i]]).view(-1).topk(self.size, 0, True, True)
                    cand.topk_cand_score = (cand_scores, cand_scores_id)
                    cand.topk_local_score = (local_scores, local_scores_id)
            elif cand.typ == 'list':
                for y in cand.y:
                    cs_beam_lk_list.append(workd_lk[i][y] + path_score.unsqueeze(0))
                y_list = cand.y
                if self.debug_flag:
                    cand.local_score=workd_lk[i][y_list]
                cand_start = cand_cnt
                cand_cnt += len(y_list)
                ks.append({'start':cand_start, 'end':cand_cnt, 'hyp_idx':cand.hyp_idx})

        if self.cs_beam_size==0:
            beam_lk_list = beam_lk_list + cs_beam_lk_list

        if len(beam_lk_list)>0:
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
                    print(probs, file=self.print_log)
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
        else:
            _scores, next_scores, next_scores_id = [], [], []

        if self.cs_beam_size>0 and len(cs_beam_lk_list)>0:
            #print('aaaaaaaaaaaaaaaaaaaaaaaa')
            cs_beam_lk = torch.cat(cs_beam_lk_list)
            cs_scores = cs_beam_lk.view(-1)
            k = min(self.cs_beam_size, cs_scores.size(0))
            cs_next_scores, cs_next_scores_id = cs_scores.topk(k, 0, True, True)
            #print(cs_next_scores_id)
            #if len(_scores)>0:
            #    print(len(_scores), _scores.shape)
            cs_next_scores_id = cs_next_scores_id + torch.tensor(len(_scores)).expand_as(cs_next_scores_id)
            #print(cs_next_scores_id)
            if len(next_scores)>0:
                #print('next_scores_id', next_scores_id)
                #print('cs_next_scores_id', cs_next_scores_id)
                next_scores = torch.cat([next_scores, cs_next_scores])
                next_scores_id = torch.cat([next_scores_id, cs_next_scores_id])
                #print('concat_next_scores_id', next_scores_id)
            else:
                next_scores = cs_next_scores
                next_scores_id = cs_next_scores_id

        for sc,idx in zip(next_scores,next_scores_id):
            if sc==-float("Inf"):
                continue
            for jj in range(len(ks)):
                if ks[jj]['end']>idx:
                    break
            cand = self.Cand[c][jj]
            ridx = idx-ks[jj]['start']
            new_filter = None
            if cand.typ=='range':
                y = cand.y[0]+ridx
                constraint = cand.cs_list[0]
                kidx = cand.kidx_list[0]
                is_open = True
                #print(cand.y, ridx, y, jj, ks[jj], sc)
                if cand.cf is not None:
                    new_filter = cand.cf.get_updated_filter(y)
            elif cand.typ=='list':
                y = torch.tensor(cand.y[ridx])
                constraint = cand.cs_list[ridx]
                kidx = cand.kidx_list[ridx]
                if cand.cf is not None:
                    new_filter = cand.cf.get_new_filter()
                #print(cand.y, ridx, y)
            new_hyp = Hyp(
                t=t, c=c,
                y=y, kidx=kidx, score=sc, pre_hyp_idx=ks[jj]['hyp_idx'],
                uc_path_len=cand.uc_path_len,
                cs=constraint,
                batch_proc_idx=jj
                )
            new_hyp.cf = new_filter
            #for dump
            new_hyp.cand_index = [jj,ridx]
            new_hyp.idx = len(new_hyps)
            ##
            new_hyps.append(new_hyp)

        # End condition is when top-of-beam is EOS.
        if new_hyps[0].y == self.eos and new_hyps[0].cs.done:
            self.done = True
        return new_hyps


    def sort_best(self, min_len=0):
        """Sort the beam."""
        #scores = torch.stack([h['score'] for h in self.Grid[self.end_t][self.numC]])
        #return torch.sort(scores, 0, True)
        self.sort_list = []
        score_list = []
        for t in range(min_len+1,BeamState.maxT): #here '1' is for eos
            for c in range(BeamState.maxC+1):
                for k, hyp in enumerate(self.Grid[t][c]):
                    #print('t=%d,c=%d:'%(t,c), hyp.cs.done, hyp.y==self.eos, hyp.uc_path_len, self.max_unconstraint_len)
                    if hyp.cs.done and (hyp.y==self.eos or hyp.uc_path_len>=self.max_unconstraint_len):
                        self.sort_list.append([t,c,k])
                        score_list.append(hyp.score)
                        #print(hyp)
        if len(score_list)==0:
            print('No done-hyps')
            return None, None
        scores = torch.stack(score_list)
        self.sorted_score, self.sorted_ks = torch.sort(scores, 0, True)
        return self.sorted_score, self.sorted_ks

    def get_sorted_path(self, k):
        """Get hypotheses."""
        hyp = []
        kidx = []
        last_t, last_c, k = self.sort_list[k]
        grid_hyps = self.Grid[last_t][last_c]
        #grid_hyps = self.Grid[self.end_t][self.numC]
        #for t in range(self.end_t-1, -1, -1):
        for t in range(last_t-1, -1, -1):
            hyp.append(grid_hyps[k].y)
            kidx.append(grid_hyps[k].kidx)
            pre_idx = grid_hyps[k].pre_hyp_idx
            #print(grid_hyps[k]['y'], self.id2word[grid_hyps[k]['y'].item()], pre_idx)
            grid_hyps = self.Grid[t][pre_idx[0]]
            k = pre_idx[1]

        return hyp[::-1],kidx[::-1]

    def dump_path(self, hyp, logger):
        hyps = []
        tokens = []
        for t in range(hyp.t-1, -1, -1):
            hyps.append(hyp)
            tokens.append(hyp.y)
            pre_idx = hyp.pre_hyp_idx
            hyp = self.Grid[t][pre_idx[0]][pre_idx[1]]
        hyps = hyps[::-1]
        tokens = tokens[::-1]
        log_txt = ''
        for i,hyp in enumerate(hyps):
            log_txt += '(%d,%d,%d)'%(hyp.t, hyp.c, hyp.idx)
            if i < (len(hyps)-1):
                if hyps[i].c == hyps[i+1].c:
                    log_txt+='='
                else:
                    log_txt+='-'
                if hyps[i].cs.isOpen:
                    log_txt+='>'
        logger.add(log_txt)
        log_txt=logger.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logger.add(log_txt)

    def dump_sorted_path(self, rank, logger):
        hyps = []
        k = self.sorted_ks[rank]
        last_t, last_c, k = self.sort_list[k]
        grid_hyps = self.Grid[last_t][last_c]
        for t in range(last_t-1, -1, -1):
            grid_hyps[k].rank.append(rank)
            hyps.append(grid_hyps[k])
            pre_idx = grid_hyps[k].pre_hyp_idx
            grid_hyps = self.Grid[t][pre_idx[0]]
            k = pre_idx[1]
        hyps = hyps[::-1]
        log_txt = ''
        for i,hyp in enumerate(hyps):
            log_txt += '(%d,%d,%d)'%(hyp.t, hyp.c, hyp.idx)
            if i < (len(hyps)-1):
                if hyps[i].c == hyps[i+1].c:
                    log_txt+='='
                else:
                    log_txt+='-'
                if hyps[i].cs.isOpen:
                    log_txt+='>'
        logger.add(log_txt)

    def dump(self, logger):
        for i in range(len(self.sorted_ks)):
            self.dump_sorted_path(i, logger)
            self.dump_sorted_path(i, logger)
        for t in range(1, BeamState.maxT):
            head_str = '======================================t=%d'%t
            logger.add(head_str)
            for c in range(BeamState.maxC+1):
                if len(self.Grid[t][c])>0:
                    head_str = '======== Grid(%d,%d):'%(t,c)
                    logger.add(head_str)

                    for i,cand in enumerate(self.GridCand[t][c]):
                        def hyp_dump_fn(j):
                            if cand.typ=='list':
                                for hyp in self.Grid[t][c]:
                                    if hyp.cand_index[0]==i and hyp.cand_index[1]==j:
                                        hyp.dump(logger)
                            else:
                                for hyp in self.Grid[t][c]:
                                    if hyp.cand_index[0]==i:
                                        hyp.dump(logger)

                        self.dump_path(self.Grid[t-1][cand.hyp_idx[0]][cand.hyp_idx[1]], logger)
                        cand.dump(t,c,hyp_dump_fn,logger) 


class GridBeamSearch(object):

    def __init__(self, beam_size, decode_fn, vocab_size, bos, eos, constrain_type='', print_log=None, debug_flag=False):
        """
        tgt_tokens: {'bos',xx, 'eos',xx}
        """
        self.beam = None
        self.decode_fn = decode_fn
        self.bos = bos
        self.eos = eos
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.constrain_type = constrain_type
        self.print_log = print_log
        self.debug_flag = debug_flag
        print('eos is %d, bos is %d'%(self.eos,self.bos))

    def beam_run(self, batch_init_states, batch_keys=None, n_best=1,
        max_unconstraint_len = 20,
        min_len = 0,
        policy = {},
        resource = {},
        repetition_penalty=1.0,
        bad_words_ids=None,
        no_repeat_ngram_size=0,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0
        ):

        #maxLen = max_length
        #if batch_keys is not None:
        #    max_numC = max([sum([len(k) for k in keys]) for keys in batch_keys if keys is not None])
        #else:
        #    maxLen = maxLen if maxLen > (max_numC+3) else (max_numC+3)

        batch_size = len(batch_init_states[0])
        if batch_keys is None:
            batch_keys = [None]*batch_size
        beam_size = self.beam_size

        #print('beam_run max_unconstraint_len ', max_unconstraint_len)
        #print('batch_size=',batch_size)
        self.beam = [
            BeamState(beam_size, max_unconstraint_len, self.vocab_size, self.bos, self.eos, [ss[k] for ss in batch_init_states],
                constrain_type=self.constrain_type, keys=batch_keys[k], policy=policy, resource=resource,
                do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                print_log=self.print_log, debug_flag=self.debug_flag)
                for k in range(batch_size)
        ]
        active_beam = [b for b in self.beam]
        #for t in range(1, maxLen):
        all_tot_hyp_cnt=0
        all_hyp_cache_cnt=0
        for t in range(1, BeamState.maxT):
            #if self.print_log:
            #    print('t=',t)
            hyp_y_set = []
            hyp_states_set = [[] for i in range(len(batch_init_states))]
            tot_hyp_cnt = 0
            remaining_sents = len(active_beam)
            #print("remaining_sents: ",remaining_sents)
            batch_start = []
            for bi,b in enumerate(active_beam):
                b.tot_hyp_cnt = 0
                batch_start.append(tot_hyp_cnt)
                #print("bi=", bi, b.numC, t, maxLen)#bi= 0 32 1 30
                #print(max(0, (b.numC+t)-maxLen), min(t, b.numC)+1)
                #for c in range(max(0, (b.numC+t)-maxLen), min(t, b.numC)+1):
                hyp_cache = {}
                hyp_index = 0
                cands_indice = []
                for c in range(0, BeamState.maxC):
                    #if self.print_log:
                    #    print('c=',c)
                    b.Cand[c] = []
                    def not_eos(y,t):
                        return not (t>1 and y==self.eos)
                    for jj,hyp in enumerate(b.Grid[t-1][c]):
                        if hyp.cs.is_open() and hyp.uc_path_len<max_unconstraint_len and not_eos(hyp.y,t):
                            if self.print_log:
                                print('generate %d'%tot_hyp_cnt, file=self.print_log)
                            hyp.cs.grp_uncs_len += 1
                            hyp_y_set.append(hyp.y)
                            for ii,s in enumerate(hyp.states):
                                hyp_states_set[ii].append(s)
                            if hyp.cf is not None:
                                if hyp.cs.inGroup:
                                    hyp.cf.generate_yfilter(hyp.cs.key.insert_vocab, hyp.cs.key.max_grp_uncs_len)
                                else:
                                    hyp.cf.generate_yfilter()
                            cand = Cand(
                                type='range', y=[0,b.vocab_size], score=hyp.score, hyp_idx=[c,jj],
                                uc_path_len=hyp.uc_path_len+1,
                                cs_list=[hyp.cs],
                                kidx_list=[-1],
                                cf=hyp.cf
                                )
                            hyp_cache[hyp] = hyp_index
                            cand.cache_i = hyp_index
                            cands_indice.append(hyp_index)
                            hyp_index += 1
                            b.Cand[c].append(cand)
                            tot_hyp_cnt += 1
                            b.tot_hyp_cnt += 1
                    if c > 0:
                        for jj,hyp in enumerate(b.Grid[t-1][c-1]):
                            if (not hyp.cs.done) and not_eos(hyp.y,t) and\
                                (hyp.cf is None or hyp.cf.is_open()):
                                #if t==12:
                                #    print('bbb')
                                y_list, cs_list, kidx_list = hyp.cs.generate_cands()
                                cand = Cand(
                                    type='list', y=y_list, score=hyp.score, hyp_idx=[c-1,jj],
                                    uc_path_len=hyp.uc_path_len,
                                    cs_list=cs_list,
                                    kidx_list=kidx_list,
                                    cf=hyp.cf
                                    )
                                if hyp not in hyp_cache:
                                    hyp_y_set.append(hyp.y)
                                    for ii,s in enumerate(hyp.states):
                                        hyp_states_set[ii].append(s)
                                    hyp_cache[hyp] = hyp_index
                                    cands_indice.append(hyp_index)
                                    cand.cache_i = hyp_index
                                    hyp_index += 1

                                else:
                                    cands_indice.append(hyp_cache[hyp])
                                    cand.cache_i = hyp_cache[hyp]

                                if self.debug_flag:
                                    cand.pre_cs=hyp.cs
                                b.Cand[c].append(cand)
                                tot_hyp_cnt += 1
                                b.tot_hyp_cnt += 1
                    if self.debug_flag:
                        b.GridCand[t][c] = b.Cand[c]
                    if self.print_log:
                        if len(b.Cand[c])>0:
                            print('++++++++ t=%d, b=%d, Cand[%d] cnt %d:'%(t,bi,c,tot_hyp_cnt), b.Cand[c], file=self.print_log)

            if tot_hyp_cnt==0:
                break
            #print(hyp_y_set, tot_hyp_cnt)
            #print(tot_hyp_cnt, len(hyp_cache))
            all_tot_hyp_cnt+=tot_hyp_cnt
            all_hyp_cache_cnt+=len(hyp_cache)
 
            input = torch.stack([y.cpu() for y in hyp_y_set]).t().contiguous().view(-1,1)

            states = [torch.stack(s) for s in hyp_states_set]
            if self.print_log:
                print('decode batch size is ', input.shape[0], file=self.print_log)
            logits, states = self.decode_fn(input, states, t)
            
            next_token_logits = logits[:,-1,:]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=states[0], #input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=t,
                min_length=min_len,
                eos_token_id=self.eos,
                repetition_penalty=repetition_penalty,
            )
            cands_indice = torch.tensor(cands_indice).to(scores.device)
            scores = scores.index_select(0, cands_indice)
            states = [s.index_select(0,cands_indice) for s in states]
            done_beam = []
            for idx,b in enumerate(active_beam):
                cand_start = 0
                task_list = []
                #for c in range(max(0, (b.numC+t)-maxLen), min(t, b.numC)+1):
                for c in range(0, BeamState.maxC):
                    if len(b.Cand[c])==0:
                        continue
                    if self.print_log:
                        print('------- t=%d, b=%d, Cand[%d] cnt %d:'%(t,idx,c,b.tot_hyp_cnt), b.Cand[c], file=self.print_log)
                    
                    if False:
                        #new_hyps = b.update(t, c, scores.data[batch_start[idx]+cand_start:batch_start[idx]+cand_start+len(b.Cand[c])].cpu())
                        new_hyps = b.double_topk_update(t, c, scores.data[batch_start[idx]+cand_start:batch_start[idx]+cand_start+len(b.Cand[c])].cpu())
                        cand_start += len(b.Cand[c])
                        for hyp in new_hyps:
                            b_idx = hyp.batch_proc_idx
                            cur_states = []
                            for i,s in enumerate(states):
                                if s is None:
                                    cur_states.append(b.Grid[0][0][0].states[i])
                                else:
                                    #shape = s.shape[1:]
                                    #cur_states.append(s.view(remaining_sents, tot_hyp_cnt, *shape)[idx,b_idx,:])
                                    cur_states.append(s[batch_start[idx]+b_idx,:])
                            hyp.states = cur_states

                        b.Grid[t][c] = new_hyps
                        b.end_t = t
                    else:
                        #if t==12:
                        #    print('aaa')
                        def update_task(tt, cc, scores_data):
                            new_hyps = b.double_topk_update(tt, cc, scores_data.cpu())
                            for hyp in new_hyps:
                                b_idx = hyp.batch_proc_idx
                                cur_states = []
                                for i,s in enumerate(states):
                                    if s is None:
                                        cur_states.append(b.Grid[0][0][0].states[i])
                                    else:
                                        #shape = s.shape[1:]
                                        #cur_states.append(s.view(remaining_sents, tot_hyp_cnt, *shape)[idx,b_idx,:])
                                        cur_states.append(s[batch_start[idx]+b_idx,:])
                                hyp.states = cur_states

                            b.Grid[tt][cc] = new_hyps

                        task_list.append(Thread(target=update_task, args=(t,c,scores.data[batch_start[idx]+cand_start:batch_start[idx]+cand_start+len(b.Cand[c])])))
                        cand_start += len(b.Cand[c])

                for tsk in task_list:
                    tsk.start()
                for tsk in task_list:
                    tsk.join()

                if b.done or b.tot_hyp_cnt==0:
                    done_beam.append(b)

            for b in done_beam:
                active_beam.remove(b)
            if len(active_beam)==0:
                break


        allHyp, allScores, allKidx = [], [], []
        #n_best = 1

        for b in range(batch_size):
            #print('b=',b)
            scores, ks = self.beam[b].sort_best(min_len)
            if scores is not None:
                allScores += [scores[:n_best]]
                #hyps = zip(*[beam[b].get_sorted_path(k)[0] for k in ks[:n_best]])
                #kidxs = zip(*[beam[b].get_sorted_path(k)[1] for k in ks[:n_best]])
                hyps = [self.beam[b].get_sorted_path(k)[0] for k in ks[:n_best]]
                kidxs = [self.beam[b].get_sorted_path(k)[1] for k in ks[:n_best]]
                allHyp += hyps
                allKidx += kidxs

        return allHyp, allKidx, allScores, all_tot_hyp_cnt, all_hyp_cache_cnt

    def dump(self, logger):
        for b in self.beam:
            b.dump(logger)

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

