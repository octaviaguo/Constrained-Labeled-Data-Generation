lang=$1
json=$2
config=../configs/$lang/$json
CUDA=$3

echo ""
CMD="python3 gbs_translate.py --config=$config --gpu_index=$CUDA"
echo ""
echo $CMD

$CMD

