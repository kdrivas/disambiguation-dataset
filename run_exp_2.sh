CUDA_IDS=$1
MAX_LENGTH=$2
TRAIN_DIR=$3
echo "Parameters " $CUDA_IDS

echo "Training 60"
python training.py --name_file=all_f1_lemma --train_dir=all_60 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_60.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_60 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_60.txt

python training.py --name_file=all_meteor_lemma --train_dir=all_60 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_60.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_60 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_60

echo "Training 70"
python training.py --name_file=all_f1_lemma --train_dir=all_70 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_70.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_70 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_70.txt

python training.py --name_file=all_meteor_lemma --train_dir=all_70 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_70.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_70 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_70


