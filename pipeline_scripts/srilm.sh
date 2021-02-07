#infolder=/shared/mayhew/nerwithpr/data/deu/Train-conll
#infolder=/shared/ruohaog/cheap_trans/data/column/eng/test
#outfolder=/shared/ruohaog/cheap_trans/data/column/deu/test
#outfolder=/shared/ruohaog/cheap_trans/debug
infolder=/shared/mayhew/ner_data/conll2003_eng/eng.train
outfolder=/shared/ruohaog/cheap_trans/new_data/de/train.conll
flist=/shared/mayhew/IdeaArchive/umt-ner/MUSE/data/crosslingual/dictionaries/en-de.txt
echo 'Input file is' $infolder
echo 'Output file is' $outfolder
python3 translate.py $infolder $outfolder $flist --file=True
