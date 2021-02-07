WIKI_LANG=nl
WIKI_DATE=20201020
PATH_DUMPDIR=/shared/ruohaog/toolkits/wikidump/dump
PATH_OUTDIR=/shared/ruohaog/toolkits/wikidump/processed

make DATE=$WIKI_DATE \
     LANG=$WIKI_LANG \
     DUMPDIR_BASE=$PATH_DUMPDIR \
     OUTDIR_BASE=$PATH_OUTDIR \
     PYTHONBIN=python all
