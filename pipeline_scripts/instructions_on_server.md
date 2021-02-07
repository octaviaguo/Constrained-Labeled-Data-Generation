# Instruction

## Wiki text

```
cd /shared/ruohaog/toolkits/wikidump/wikidump-preprocessing (48min for Spanish)
conda activate old
vim run.sh   #Change "WIKI_LANG""WIKI_DATE";
sh run.sh
```

```
cd /shared/ruohaog/toolkits
conda activate old
python3 wiki2text.py ##Change
conda activate allennlp
python3 sbd.py  ##Change
```
