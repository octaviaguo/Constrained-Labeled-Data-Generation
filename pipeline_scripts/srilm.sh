infolder=$SOURCE_DATA_FILE
outfolder=$TARGET_DATA_FILE
flist=$DICTIONARY_FILE
echo 'Input file is' $infolder
echo 'Output file is' $outfolder

python3 translate.py $infolder $outfolder $flist --file=True
