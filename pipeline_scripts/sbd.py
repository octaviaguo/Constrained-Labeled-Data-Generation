# sentence boundary detector
import spacy
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True, help="the language to process")
parser.add_argument("--input", type=str, required=True, help="input file")
parser.add_argument("--output", type=str, required=True, help="output file")
parser.add_argument("--doc_thrd", type=int, required=True, help="the max number of documents to process")
parser.add_argument("--lm", type=str, required=True, help="pretrained spacy language model for sentence tokenization")
args = parser.parse_args()

lang = args.lang
inpath = args.input
outpath = args.output
doc_threshold = args.doc_thrd
nlp = spacy.load(args.lm)

start_doc = 0
doc_index = 0
count = 0
sent_num = 0
no_empty_before = True

with open(inpath, "r") as infile:
    while True:
        if doc_index < start_doc:
            doc_index += 1
            line = infile.readline()
            continue
        
        line = infile.readline()

        if not line:
            break

        line = line.strip()
        
        if ((not line) and count) or ('ISO' in line and len(line.split())<4): # empty line
            if no_empty_before:
                with open(outpath,"a") as outfile:
                    outfile.write('\n')
                no_empty_before = False
            continue

        no_empty_before = True
        
        doc = nlp(line)
        with open(outpath, "a") as outfile:
            for sent in doc.sents:
                #print(sent.text)
                sent_num += 1
                outfile.write(sent.text +'\n')
        
        count += 1
        if count%1000==0:
            print("Processed\t"+str(count+start_doc))

print("# of documents: ", count-1)
print("# of sentences: ", sent_num)

