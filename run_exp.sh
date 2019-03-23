CUDA_IDS=$1
MAX_LENGTH=$2
TRAIN_DIR=$3
echo "Parameters " $CUDA_IDS

python training.py --name_file=all_f1_lemma --train_dir=$TRAIN_DIR --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1.txt
python training.py --name_file=all_f1_random_lemma --train_dir=$TRAIN_DIR --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random.txt

python training.py --name_file=all_meteor_lemma --train_dir=$TRAIN_DIR --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=$TRAIN_DIR --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random.txt

