CUDA_IDS=$1
MAX_LENGTH=$2
echo "Parameters " $CUDA_IDS

echo "Training 50"
python training.py --name_file=all_f1_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_50.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_50.txt

python training.py --name_file=all_meteor_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_50.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_50.txt


