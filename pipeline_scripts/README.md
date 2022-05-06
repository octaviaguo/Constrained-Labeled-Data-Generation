# Pipeline of Constrained Labeled Data Generation for Low-Resource Named Entity Recognition

## Instructions 

### Update [Xie et al.(2018)](https://aclanthology.org/D18-1034.pdf)'s translation code
```
git clone https://github.com/thespectrewithin/cross-lingual_NER.git
vi run_transfer_training_data.sh
# change "WORD_TRANSLATION_FILE": path to 2-col dict
# change "SOURCE_TRAINING_DATA": path to 2-col eng train data
# change "OUTPUT_FILE" : path to output translated data

sh run_transfer_training_data.sh
```

### Get Wiki text
#### Extract from [Wikidump](https://github.com/CogComp/wikidump-preprocessing)
```
git clone https://github.com/CogComp/wikidump-preprocessing.git
cd wikidump-preprocessing
#pip install following the "Requirements" in README.md

#then go back to .../Constrained-Labeled-Data-Generation/pipeline_scripts/
vim extract_wikidump.sh #Change "WIKI_LANG", "WIKI_DATE", "PATH_DUMPDIR", "PATH_OUTDIR"(the directory for extracted texts)
sh extract_wikidump.sh
```
#### Process Wikidump results to get text (1 sentence/line) 
```
python3 wiki2text.py --lang {language} --base {folder of the processed Wikidump} --output {output_file}
#Output is text, 1 doc/line

 
python3 sbd.py --lang {language} --input {input_file} --output {output_file} --doc_thrd {max_num_of_documents} --lm {spacy_language_model}
#Output is text, 1 sentence/line.
```

### Bart Training
#### Bart input: tfidf  
```
cd ../key_words_extract/data
mkdir {language_folder}  #mkdir a folder to save tfidf.kw
python3 tfidf_extract.py --train_data=train_data_file --dev_data=development_data_file --rate=0.25
```
#### Train
Follow the "Prepare data" and "Train model" in this [site](https://github.com/octaviaguo/Constrained-Labeled-Data-Generation/tree/main/t5_train). Specifically, you need to: 
* tokenize input
* write config file
* train a language model (python3 t5_train.py --config=...)

### Cheap Translation (select best translation in dictionay with SRILM)
#### Train ngram for the target language:
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

#### Start SRILM Server from another terminal
cd .../SRILM/lm/bin/i686-m64
./ngram -lm {lm} -server-port 8181

#### Translate
```
git clone https://github.com/CogComp/python-conll.git
cd python-conll/bin
## copy "srilm.sh" "my_translate.py" into the current place; 
## The difference between "my_translate.py" and the github "translate.py" is: "my_translate" supports translating files, directory; "translate" translate only directory

#then go back to .../Constrained-Labeled-Data-Generation/pipeline_scripts/
vim srilm.sh #Change infolder(src to translate), outfolder, flist(the dictionary)
sh srilm.sh
```



### CLDG Translation
Define your translation config json file under ```.../Constrained-Labeled-Data-Generation/configs/{lang}/```.

Then go back here to run:

```

sh gbs_trans.sh {lang} {config.json} {GPU_index}
```
