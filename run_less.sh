CUDA_IDS=$1
MAX_LENGTH=$2
echo "Parameters " $CUDA_IDS

echo "Training with f1"
python training.py --name_file=all_f1_lemma --train_dir=all_less --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_eq.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_less --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_eq.txt

echo "Training with meteor"
python training.py --name_file=all_meteor_lemma --train_dir=all_less --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_eq.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_less --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_eq.t$



