
import spacy
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True, help="the language to process")
parser.add_argument("--input", type=str, required=True, help="input file")
parser.add_argument("--output", type=str, required=True, help="output file")
args = parser.parse_args()

lang = args.lang
inpath = args.input
outpath = args.output
count = 0
sent_num = 0
printsample = 1000
min_sent_len = 6

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


with open(inpath, "r") as infile:
    while True:
        line = infile.readline()
        if not line:
            break    

        sents = split_into_sentences(line)

        if len(sents):
            for sent in sents:
                if sent:
                    if sent=='\n' or len(sent.split()) < min_sent_len:
                        continue
                    if "http" in sent:
                        continue
                        
                    with open(outpath, "a") as outfile:
                        outfile.write(sent+ '\n')
                        sent_num += 1
            count += 1
        
            if count % printsample==0:
                print("Process # of sents\t", sent_num)




print("# of documents: ", count-1)
print("# of sentences: ", sent_num)

