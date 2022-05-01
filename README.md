##  ACT

Code for our NAACL-HLT 2022 paper: "[Neighbors Are Not Strangers: Improving Non-Autoregressive Translation under Low-Frequency Lexical Constraints](https://arxiv.org/abs/2204.13355)".

### Requirements

- transformers>=4.2.0
- python version>=3.6
- pytorch version>=1.2.0

### Preparation

#### Command

- Enter the fairseq folder, run the command

  ```bash
  pip install --editable ./
  ```

#### File

- Language pair dataset: Binarized language files.
- Test set constraint: each line represents a constraint for the corresponding language pair in the format "4|immer", representing that the fourth word of the source language word needs to be translated to "immer".


- Alignment file: First use the GIZA++ tools to get alignment file, then put them into the binarized folder and name it "train.align_giza" and "valid.align_giza".
- TF-IDF file: Put them into the binarized folder and name it "train.tfidf.score", "train.tfidf.word", "valid.tfidf.score", "valid.tfidf.word", "test.tfidf.score", "test.tfidf.word". We build the tfidf by gensim.

### Train

```bash
python train.py data/wmt14-en-de-distill-bin \
--save-dir checkpoints/wmt14_ende_distill_ACT \
--ddp-backend=legacy_ddp --task translation_lev \
--criterion nat_loss --arch levenshtein_transformer \
--noise random_delete_wo_cs \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 \
--warmup-updates 10000 --warmup-init-lr 1e-07 \
--label-smoothing 0.1 --dropout 0.3 --weight-decay 0.01 \
--decoder-learned-pos --encoder-learned-pos --apply-bert-init \
--log-format simple --log-interval 1 --fixed-validation-seed 7 \
--max-tokens 8000 --save-interval-updates 10000 --max-update 300000 \
--source-lang en --target-lang de \
--keep-best-checkpoints 5 \
--update-freq 4 \
--with-multitask 0 --fp16
```

### Average the checkpoints

```bash
python ../fairseq/scripts/average_checkpoints.py \
--inputs checkpoints/wmt14_ende_distill_ACT \
--num-epoch-checkpoints  5 --output checkpoint_aver.pt
```

### Generate

we can set the decoding mode 0 for no constraint, 1 for soft constraint and 2 for hard constraint

```bash
python generate.py data/wmt14-en-de-distill-bin \
--gen-subset test --task translation_lev \
--path checkpoint_aver.pt \
--iter-decode-max-iter 9 --iter-decode-eos-penalty 0 \
--beam 1 --remove-bpe --print-step \
--batch-size 400 --decoding-mode 0 \
--bpe subword_nmt --tokenizer moses \
--source-lang en --target-lang de \
--bpe-codes data/wmt14-en-de-distill-bin/wmt14.bpe.codes
```


## Citation

If you find our paper useful to your work, please kindly cite our paper:
```latex
@misc{zeng2021neighbors,
      title={Neighbors Are Not Strangers: Improving Non-Autoregressive Translation under Low-Frequency Lexical Constraints},
      author={Zeng, Chun and Chen, Jiangjie and Zhuang, Tianyi and Xu, Rui and Yang, Hao and Qin, Ying and Tao, Shimin and Xiao, Yanghua},
      year={2022},
      eprint={2204.13355},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

