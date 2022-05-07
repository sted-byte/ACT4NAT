# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
from dataclasses import dataclass, field
import torch
import os
import json
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange
from tqdm import tqdm
import pickle
import numpy as np

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask", "random_delete_wo_cs"])

@dataclass
class TranslationLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={
            "help": "type of noise"
        },
    )

@register_task("translation_lev", dataclass=TranslationLevenshteinConfig)
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationLevenshteinConfig

    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg,src_dict,tgt_dict)
        self.constraint_training=True
        self.multi_task = False
        self.constraints={"train":None,"valid":None,"test":None}
        self.alignment={"train":None,"valid":None,"test":None}
        self.tfidf_word={"train":None,"valid":None,"test":None}
        self.tfidf_score={"train":None,"valid":None,"test":None}
        self.test_cs_pos_path=os.path.join(self.cfg.data,"test.cs_pos")
        self.giza = True
        self.additional_str=""
        # if self.constraint_training:
            # self.build_constraints("train")
            # self.build_constraints("test")
            # self.build_aligment("train")
            # self.build_aligment("valid")        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.build_aligment("test")       // For test, we use test.cs_pos

    def build_tfidf(self,dataset):
        self.tfidf_word[dataset]=[]
        self.tfidf_score[dataset]=[]
        path_word=os.path.join(self.cfg.data,dataset+".tfidf.word")
        path_score=os.path.join(self.cfg.data,dataset+".tfidf.score")
        with open(path_word,"r",encoding="UTF-8") as f_in:
            data_word=[_.strip() for _ in f_in.readlines()]
        with open(path_score,"r",encoding="UTF-8") as f_in:
            data_score=[_.strip() for _ in f_in.readlines()]
        for i in tqdm(range(len(data_word))):
            self.tfidf_word[dataset].append(json.loads(data_word[i]))
            self.tfidf_score[dataset].append(json.loads(data_score[i]))

    def build_aligment(self,dataset):
        if self.giza:
            apd = "_giza"
        else:
            apd = ""
        print("Building {} alignment".format(dataset))
        path=os.path.join(self.cfg.data,dataset+".align"+apd)
        cache_path = os.path.join("cache","{}.align.cache".format(dataset+self.cfg.source_lang+"-"+self.cfg.target_lang)+apd)
        print("check the path {}".format(cache_path))
        # if not os.path.exists("cache"):
        #     os.mkdir("cache")
        # if os.path.exists(cache_path):
        #     self.alignment[dataset] = self.load_pickle(cache_path)
        #     return
        self.alignment[dataset]=[]
        with open(path,"r",encoding="UTF-8") as f_in:
            data=[_.strip().split(" ") for _ in f_in.readlines()]
        for i in tqdm(range(len(data))):
            temp={}
            if len(data[i][0]) != 0:
                for j in range(len(data[i])):
                    a=data[i][j].split("-")[0]
                    b=data[i][j].split("-")[1]
                    if b not in temp:
                        temp[b]=[a]
                else:
                    temp[b].append(a)
            self.alignment[dataset].append(temp)
        # self.save_pickle(self.alignment[dataset],cache_path)
    # Add by zc
    def build_constraints(self,dataset):
        if len(self.additional_str)>0:
            path = self.additional_str
        else:
            path=os.path.join(self.cfg.data,dataset+".constraints")
        with open(path,"r",encoding="UTF-8") as f_in:
            data=[_.strip() for _ in f_in.readlines()]
            data=[_.split("\t") if len(_)>0 else[] for _ in data]
        self.constraints[dataset]=data


    def load_pickle(self, path):
        print("读取pickle")
        with open(path, 'rb') as fil:
            data = pickle.load(fil)
        return data

    def save_pickle(self, en, path):
        print("保存pickle")
        with open(path, 'wb') as fil:
            pickle.dump(en, fil)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
            small_dataset=kwargs['small_dataset'] if 'small_dataset' in kwargs.keys() else 0,
        )

    def inject_noise(self, target_tokens, constraints_mask = None, ratio = 1):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        def _random_delete_wo_cs(target_tokens, constraints_mask, ratio):
            assert constraints_mask != None
            constraints_mask=constraints_mask.to(target_tokens.device)
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)

            constraints_len = constraints_mask.sum(1, keepdim=True)

            target_score.masked_fill_(constraints_mask.bool(),0.0)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                      2 + constraints_len
                    + (
                            (target_length -2 - constraints_len)
                            * target_score.new_zeros(target_score.size(0), 1).uniform_()
                    ).long()
            )

            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                    .masked_fill_(target_cutoff, pad)
                    .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                                 :, : prev_target_tokens.ne(pad).sum(1).max()
                                 ]

            return prev_target_tokens

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "random_delete_wo_cs":
            if constraints_mask == None:
                return _random_delete(target_tokens)
            else:
                return _random_delete_wo_cs(target_tokens,constraints_mask, ratio)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def get_constraint_mask(self, target, constraints, dict):
        constraints=[[[dict.indices[laus] for laus in tp.split(" ")] for tp in _] for _ in constraints]
        constraints_mask=torch.zeros(target.shape)
        list_target=[[laus.item() for laus in _] for _ in list(target)]
        for i in range(0,len(list_target)):
            end_index=list_target[i].index(2)
            ready=[1 for _ in range(len(constraints[i]))]
            for j in range(len(list_target[i])):
                if sum(ready) == 0:
                    break
                for k in range(len(constraints[i])):
                    if not ready[k]:
                        continue
                    if j+len(constraints[i][k])-1 < end_index and list_target[i][j:j+len(constraints[i][k])] == constraints[i][k]:
                        constraints_mask[i][j:j+len(constraints[i][k])] = True
                        ready[k] = 0
            assert sum(ready) == 0
        return constraints_mask

        # Add by zty

    def get_constraint_mask_dynamic(self, target, constraints, training_step, max_update):
        constraints = [[[self.tgt_dict.indices[laus] for laus in tp.split(" ")] for tp in _] for _ in constraints]
        constraints_mask = torch.zeros(target.shape)
        list_target = [[laus.item() for laus in _] for _ in list(target)]
        total_steps = max_update or 3e5
        for i in range(0, len(list_target)):
            end_index = list_target[i].index(2)
            ready = [1 for _ in range(len(constraints[i]))]
            constraint_count = 0
            for j in range(len(list_target[i])):
                if sum(ready) == 0:
                    break
                if training_step > 0.8 * total_steps and constraint_count > 0:
                    break
                if training_step > 0.5 * total_steps and constraint_count > 1:
                    break
                for k in range(len(constraints[i])):
                    if not ready[k]:
                        continue
                    if j + len(constraints[i][k]) - 1 < end_index and list_target[i][j:j + len(constraints[i][k])] == \
                            constraints[i][k]:
                        constraint_count += 1
                        constraints_mask[i][j:j + len(constraints[i][k])] = constraint_count
                        ready[k] = 0
            assert (sum(ready) == 0 or training_step > 0.5 * total_steps)
        # something be like [0001100022...]
        return constraints_mask

    def get_random_constraint(self,str_infos_tgt,str_infos_src,align,probability,prob_cnt,src_tokens):
        constrained_word_nums=np.random.choice([0,1,2,3],len(str_infos_tgt['tokens']),p=[0.4,0.3,0.2,0.1])
        indicator=torch.zeros(len(str_infos_src['tokens']),len(str_infos_src['tokens'][0])+2).int()
        result=[[] for _ in range(len(str_infos_tgt['tokens']))]
        for i in range(0,len(str_infos_tgt['indexs'])):
            if prob_cnt[i]==0:
                continue
            words_id = np.sort(np.random.choice(len(str_infos_tgt['indexs'][i]),
                                                min(min(constrained_word_nums[i], len(str_infos_tgt['indexs'][i])),
                                                    prob_cnt[i]), replace=False, p=probability[i]))
            cnt=1
            padding_add=sum(src_tokens[i]==1).item()
            for word_id in words_id:
                if str(word_id) not in align[i]:
                    continue
                tp=int(align[i][str(word_id)][0])
                # if tp >= len(str_infos_src['indexs'][i]):
                #     continue
                src_tp=str_infos_src['indexs'][i][int(align[i][str(word_id)][0])]
                indicator[i][torch.tensor(src_tp)+1+padding_add]=cnt
                cnt+=1
                tp=str_infos_tgt['indexs'][i][word_id]
                constraint_word=" ".join(str_infos_tgt['tokens'][i][_] for _ in tp)

                result[i].append(constraint_word)
        indicator=torch.tensor(indicator).int()
        return result,indicator

    def get_str_infos(self,target,tgt_dict):
        tokens=[[tgt_dict.symbols[laus.item()] for laus in _[_.ne(0)&_.ne(2)&_.ne(1)]] for _ in target]
        indexs=[[] for i in range(len(tokens))]
        for i in range(len(tokens)):
            j=0
            while j <len(tokens[i]):
                indexs[i].append([j])
                while tokens[i][j].endswith("@@") and j + 1 < len(tokens[i]):
                    while tokens[i][j].endswith("@@") and j + 1 < len(tokens[i]):
                        indexs[i][-1].append(j + 1)
                        j += 1
                    while j + 2 < len(tokens[i]) and tokens[i][j + 1] == "@-@":
                        indexs[i][-1].extend([j + 1, j + 2])
                        j += 2
                while j + 2 < len(tokens[i]) and tokens[i][j + 1] == "@-@":
                    while j + 2 < len(tokens[i]) and tokens[i][j + 1] == "@-@":
                        indexs[i][-1].extend([j + 1, j + 2])
                        j += 2
                    while tokens[i][j].endswith("@@") and j + 1 < len(tokens[i]):
                        indexs[i][-1].append(j + 1)
                        j += 1
                j+=1
        str_infos={}
        str_infos["tokens"]=tokens
        str_infos["indexs"]=indexs
        return str_infos

    def get_probability(self, str_infos, align, tfidf_word):
        probability = []
        prob_cnt = []
        punc = [',', '.', '?', '@-@', '-']
        for i in range(len(str_infos['indexs'])):
            now = np.array([False if " ".join(
                str_infos['tokens'][i][str_infos['indexs'][i][_][0]:str_infos['indexs'][i][_][-1] + 1]).replace("@@ ",
                                                                                                                "").replace(" @-@ ","-") not in punc
                            and str(_) in align[i]
                            and " ".join(
                str_infos['tokens'][i][str_infos['indexs'][i][_][0]:str_infos['indexs'][i][_][-1] + 1]).replace("@@ ",
                                                                                                                "").replace(" @-@ ","-") in tfidf_word[i][:5]
                            else True
                            for _ in range(len(str_infos['indexs'][i]))])
            cnt = len(str_infos['indexs'][i]) - now.sum()
            prob_cnt.append(cnt)
            now_prob = np.array([1 / cnt if cnt > 0 else 0 for _ in range(len(str_infos['indexs'][i]))])
            now_prob[now] = 0
            probability.append(list(now_prob))
        return probability, prob_cnt

    def build_bert_input(self, src_tokens, src_dict):
        inputs=src_tokens.cpu().clone()
        real = inputs.ne(src_dict.pad_index)
        masked_indices=torch.bernoulli(torch.full(src_tokens.shape,0.15)).bool()
        indices_replaced=torch.bernoulli(torch.full(src_tokens.shape,0.8)).bool()&masked_indices&real
        inputs[indices_replaced]=src_dict.pad_index

        indices_random = torch.bernoulli(torch.full(src_tokens.shape, 0.5)).bool() & masked_indices & ~indices_replaced & real
        random_words = torch.randint(len(src_dict), src_tokens.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        encoder_mlm_mask = indices_replaced | indices_random
        return inputs.to(src_tokens.device), encoder_mlm_mask.to(src_tokens.device);

    def train_step(
        self, sample, model, criterion, optimizer, update_num, max_update, ignore_grad=False
    ):
        model.train()
        if "constraints" not in sample.keys():
            if self.constraint_training:
                # for random sampling
                if "str_infos_tgt" not in sample.keys():
                    sample["str_infos_tgt"]=self.get_str_infos(sample["target"],self.tgt_dict)
                    sample["str_infos_src"]=self.get_str_infos(sample["net_input"]["src_tokens"],self.src_dict)
                if "probability" not in sample.keys():
                    sample['probability'],sample['prob_cnt'] = self.get_probability(sample['str_infos_tgt'], [self.alignment["train"][_.item()] for _ in sample['id']], [self.tfidf_word['train'][_.item()] for _ in sample['id']])
                sample["constraints"],sample["indicator"] = self.get_random_constraint(sample["str_infos_tgt"],sample["str_infos_src"],[self.alignment["train"][_.item()] for _ in sample['id']],sample['probability'],sample['prob_cnt'],sample['net_input']['src_tokens'])
                sample["indicator"]=sample["indicator"].to(sample["target"].device)
                sample["constraints_mask"] = self.get_constraint_mask(sample["target"], sample["constraints"], model.tgt_dict)
            else:
                sample["constraints"] = None
                sample["constraints_mask"] = None
        sample["prev_target"] = self.inject_noise(sample["target"],sample["constraints_mask"])
        if self.multi_task:
            sample["src_tokens_after"], sample["encoder_mlm_mask"] = self.build_bert_input(sample["net_input"]["src_tokens"], self.src_dict)
        else:
            sample["src_tokens_after"] = None
            sample["encoder_mlm_mask"] = None
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if "constraints" not in sample.keys():
                if self.constraint_training:
                    if "str_infos_tgt" not in sample.keys():
                        sample["str_infos_tgt"] = self.get_str_infos(sample["target"], self.tgt_dict)
                        sample["str_infos_src"] = self.get_str_infos(sample["net_input"]["src_tokens"], self.src_dict)
                    if "probability" not in sample.keys():
                        sample['probability'], sample['prob_cnt'] = self.get_probability(sample['str_infos_tgt'], [self.alignment["valid"][_.item()] for _ in sample['id']], [self.tfidf_word['train'][_.item()] for _ in sample['id']])
                    sample["constraints"], sample["indicator"] = self.get_random_constraint(sample["str_infos_tgt"],
                                                                                            sample["str_infos_src"], [
                                                                                                self.alignment["valid"][
                                                                                                    _.item()] for _ in
                                                                                                sample['id']],
                                                                                            sample['probability'],
                                                                                            sample['prob_cnt'],
                                                                                            sample['net_input'][
                                                                                                'src_tokens'])
                    sample["indicator"] = sample["indicator"].to(sample["target"].device)
                    sample["constraints_mask"] = self.get_constraint_mask(sample["target"], sample["constraints"],
                                                                          model.tgt_dict)
            sample["prev_target"] = self.inject_noise(sample["target"],sample["constraints_mask"])
            if self.multi_task:
                sample["src_tokens_after"], sample["encoder_mlm_mask"] = self.build_bert_input(sample["net_input"]["src_tokens"], self.src_dict)
            else:
                sample["src_tokens_after"] = None
                sample["encoder_mlm_mask"] = None
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
