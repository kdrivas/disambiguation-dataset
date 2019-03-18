python training.py --name_file=all_lemma --cuda=True --max_length=100 --cuda_ids=[0,3] > all.txt

python training.py --name_file=all_f1_lemma --cuda=True --max_length=100 --cuda_ids=[0,3] > f1.txt
python training.py --name_file=all_f1_random_lemma --cuda=True --max_length=100 --cuda_ids=[0,3] > f1_random.txt

python training.py --name_file=all_meteor_lemma --cuda=True --max_length=100 --cuda_ids=[0,3] > meteor.txt
python training.py --name_file=all_meteor_random_lemma --cuda=True --max_length=100 --cuda_ids=[0,3] > meteor_random.txt

