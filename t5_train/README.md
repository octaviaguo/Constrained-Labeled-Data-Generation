# environment setup
1. conda create -n pytorch1.6
2. conda activate pytorch1.6
3. conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# get transformer code
1. cd Better-Cheap-Translation/
2. git clone https://github.com/huggingface/transformers.git (in the project root directory of Better-Cheap-Translation)

# prepare data
1. put train data in path of "../data/en-10w"
2. cd t5_train
## (for t5)
3. mkdir tokenized_data
4. python3 ./train_sentencepiece.py -i ../data/en-10w/original.train -m tokenized_data/spiece 
## (for bart)
3. python3 bpe_tokenize.py -i ../data/en-10w/original.train -m bart_tokenized_data


# train model
1. cd t5_train
2. mkdir log
## (for t5)
3. python3 t5_train.py --config=myConfig.json --gpu_index=0 \
python3 t5_train.py --config=myConfig.json --gpu_index=0 --load_epoch=10
## (for bart)
3. python3 t5_train.py --config=myBartConfig.json --gpu_index=0

# test model
1. cd t5_train
## (for t5)
2. examples \
python3 t5_train.py --config=myConfig.json --type=test --load_epoch=30 \
python3 t5_train.py --config=t5keyConfig.json --type=test --model=t5_models/epoch_94.ckpt  --test_batch_size=1 --num_beams=4
## (for bart)
2. python3 t5_train.py --config=myBartConfig.json --type=test --load_epoch=30 \
python3 t5_train.py --config=configs/bart-base_10w_0.25_grid_beam_Config.json --type=test_grid_beam --num_beams=4 --model=../models/facebook-bart-base_10w-bart-0.25_epoch_80.ckpt \
python3 t5_train.py --config=configs/bart-base_10w_0.25_grid_beam_Config.json --type=test_grid_beam --num_beams=4 --do_sample=True --model=../models/facebook-bart-base_10w-bart-0.25_epoch_80.ckpt \

# test pretrained and self trained model
## (for t5)
1. python3 t5_test.py --model=epoch_8.ckpt --input="Hello, I lived in USA."
## (for bart)
1. python3 bart_test.py

#config json:
1. model type:\
  	t5-small: 6 layers for both encoder and decoder\
  	t5-base:  12 layers \
  	t5-large: 24 layer\
  	t5-3b\
  	t5-lib\
  	facebook/bart-base: 6 layers for both encoder and decoder\
  	facebook/bart-large: 12 layers\
  	facebook/bart-large-mnli\
  	facebook/bart-large-cnn\
  	facebook/bart-large-xsum\
  	facebook/mbart-large-en-ro\
  	yjernite/bart_eli5

2. training type:\
	from-scratch\
	fine-tune
