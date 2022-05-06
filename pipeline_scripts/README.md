Pipeline of training NER with Better Cheap Translation for low-resource languages

# Instructions 

## Update [Xie et al.(2018)](https://aclanthology.org/D18-1034.pdf)'s translation code
```
git clone https://github.com/thespectrewithin/cross-lingual_NER.git

vi run_transfer_training_data.sh
# change "WORD_TRANSLATION_FILE": path to 2-col dict
# change "SOURCE_TRAINING_DATA": path to 2-col eng train data
# change "OUTPUT_FILE" : path to output translated data

sh run_transfer_training_data.sh

```


## Get Wiki text

### Extract from [Wikidump](https://github.com/CogComp/wikidump-preprocessing)
```
git clone https://github.com/CogComp/wikidump-preprocessing.git
cd wikidump-preprocessing
#pip install following the "Requirements" in README.md
vim extract_wikidump.sh #Change "WIKI_LANG","WIKI_DATE", "PATH_DUMPDIR"(does not use this), "PATH_OUTDIR"(the directory for extracted texts)
sh extract_wikidump.sh
```
### Process Wikidump results to get text (1 sentence/line) 
```
cd /shared/ruohaog/toolkits (Only on Server)
vim wiki2text.py; #Change: lang, base, outpath, fold_threshold
conda activate old 
python3 wiki2text.py #Output is text, 1 doc/line
conda activate allennlp
vim sbd.py  #Use spacy to do sentence tokenization (needs to download pretrained spacy lm for each language; change line: nlp = spacy.load("de_core_news_sm")  )
python3 sbd.py #Output is text, 1 sentence/line.
```

## Bart Training
### Bart input: tfidf  
```
conda activate t5
cd /shared/ruohaog/BCT/together/backup/Better-Cheap-Translation/key_words_extract/
cd data  #mkdir a folder to save tfidf.kw
python3 tfidf_extract.py --train_data=/shared/ruohaog/wiki_data/de/train.txt --dev_data=/shared/ruohaog/wiki_data/de/dev.txt --rate=0.25
```
### Train
Following the "Prepare data" and "Train model" in this [site](https://github.com/octaviaguo/Better-Cheap-Translation/tree/main/t5_train). Specifically, you need to: 
* tokenize input
* write config file
* train (python3 t5_train.py --config=c...)

## Cheap Translation (select best translation in dictionay with SRILM)
### Paper code
I just found a [github](https://github.com/mayhewsw/python-translate)
### Train ngram for the target language:
```
git clone https://github.com/BitSpeech/SRILM.git
cd SRILM
```
See **INSTALL** for build instructions. Then:
```
cd lm/bin/i686-m64
./ngram-count -text {text} -lm {lm} -tolower -unk #{text}:wiki text(1 sentence/line), {lm} name of SRILM language model to save.
```
More details of SRILM command [here](http://www.speech.sri.com/projects/srilm/manpages/ngram.1.html).


### Start SRILM Server from another terminal
cd .../SRILM/lm/bin/i686-m64
./ngram -lm {lm} -server-port 8181

### Translate
```
git clone https://github.com/CogComp/python-conll.git
cd python-conll/bin
## copy "srilm.sh" "my_translate.py" into the current place; 
## The difference between "my_translate.py" and the github "translate.py" is: "my_translate" supports translating files, directory; "translate" translate only directory
vim srilm.sh #Change infolder(src to translate), outfolder, flist(the dictionary)
sh srilm.sh
```

## Improve cheap translation with pretrained Bart
### Prepare input, NER keys for Bart
```
sh run_process_ct.sh   #change variables
```
### Bart grid-beam-search; output in CoNLL03 format, IOB1
First, modify config file. Then:
```
sh run_gbs.sh   #change variables
```

## NER Training

## One step translation

```
python3 gbs_translate.py --config=../configs/de/ccc.json
the src_path is configured with data.default_src in ccc.json, the output file is put in directory of ./data/lll (here lll is data.target_lan configured in xxx.json)

more examples:
below two examples use better dict, input is 9 col format:
python3 gbs_translate.py --config=../configs/gbs_hard_constrain_CheapTranInput_config.json --src_path=../tmp/001.txt --output_file=./bbb.txt --log_file=./log.txt
python3 gbs_translate.py --config=../configs/gbs_dict_constrain_CheapTranInput_config.json --src_path=../tmp/001.txt --output_file=./bbb.txt --log_file=./log.txt
below two examples use simple dict, for test only:
python3 gbs_translate.py --config=../configs/gbs_hard_constrain_config_example.json --src_path=../tmp --output_file=./aaa.txt
python3 gbs_translate.py --config=../configs/gbs_dict_constrain_config_example.json --src_format=col4 --src_path=../tmp/eng.train --output_file=./aaa.txt

```
