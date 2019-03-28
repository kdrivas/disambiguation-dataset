CUDA_IDS=$1
MAX_LENGTH=$2
echo "Parameters " $CUDA_IDS

echo "Training 30"
echo "Training with f1"
python training.py --name_file=all_f1_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_30.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_30.txt

echo "Training with meteor"
python training.py --name_file=all_meteor_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_30.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_30


