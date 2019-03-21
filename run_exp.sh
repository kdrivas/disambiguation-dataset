CUDA_IDS=$1
MAX_LENGTH=200

python training.py --name_file=all_lemma --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> all.txt

python training.py --name_file=all_f1_lemma --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS & > f1.txt
python training.py --name_file=all_f1_random_lemma --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random.txt

python training.py --name_file=all_meteor_lemma --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor.txt
python training.py --name_file=all_meteor_random_lemma --cuda=True --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random.txt

