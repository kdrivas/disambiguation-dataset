CUDA_IDS=$1
MAX_LENGTH=$2
TRAIN_DIR=$3
echo "Parameters " $CUDA_IDS

python training.py --name_file=all_f1_lemma --train_dir=all_20 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_20.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_20 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_20.txt

python training.py --name_file=all_meteor_lemma --train_dir=all_20 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_20.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_20 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_20.txt

echo "Training 30"
python training.py --name_file=all_f1_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_30.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_30.txt

python training.py --name_file=all_meteor_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_30.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_30 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_30

echo "Training 40"
python training.py --name_file=all_f1_lemma --train_dir=all_40 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_40.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_40 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_40.txt

python training.py --name_file=all_meteor_lemma --train_dir=all_40 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_40.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_40 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_40

echo "Training 50"
python training.py --name_file=all_f1_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> f1_50.txt
python training.py --name_file=all_f1_random_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS  &> f1_random_50.txt

python training.py --name_file=all_meteor_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_50.txt
python training.py --name_file=all_meteor_random_lemma --train_dir=all_50 --max_length=$MAX_LENGTH --cuda_ids=$CUDA_IDS &> meteor_random_50

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



