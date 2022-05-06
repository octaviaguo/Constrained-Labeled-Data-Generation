from os import listdir
from os.path import isfile, join
import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True, help="the language to process")
parser.add_argument("--base", type=str, required=True, help="folder where the processed Wikidump is stored")
parser.add_argument("--output", type=str, required=True, help="where to store the output file")
parser.add_argument("--min_len", type=int, default=6, help="minimum length of text to be considered")
args = parser.parse_args()

lang = args.lang
base = args.base
outpath = args.output
min_w = args.min_len

folds=[f for f in listdir(base)]
allFiles = []
count = 0
for fold in folds:
    count += 1
    inpath = base + fold
    allFiles = allFiles + [join(inpath, f) for f in listdir(inpath) if isfile(join(inpath, f)) and ".brief" in join(inpath, f)]
print("# of actual folders:  ", count)
print("# of Wiki files: ", len(allFiles))

print_sample = 20
for idx, fpath in enumerate(allFiles):  
    if idx%print_sample==0:
        print(fpath)
        
    with open(fpath, "r") as infile:
        doc_collection = json.load(infile)
        for doc in doc_collection:
            raw = re.sub(r'\n+', '\n', doc["text"])
            if len(raw.strip().split()) > min_w:
                with open(outpath, "a") as outfile:
                    outfile.write(raw)

print("\ndone")
