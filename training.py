# coding: utf-8

import itertools
from typing import Iterator, List, Dict
import fire

from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.fields import TextField, IndexField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance

from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.models import DecomposableAttention
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.util import  prepare_environment
from allennlp.common.params import  Params

prepare_environment(Params({'random_seed': 123,
                                  'numpy_seed' : 123,
                                  'pytorch_seed' : 123}))

from src.data import prepare_data
from src.utils import get_stats
import numpy as np
import os 
from pathlib import Path

EN_EMBEDDING_DIM = 256
ZH_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = 0

import torch.optim as optim
import torch

def evaluate_acc(predictor, test_dataset, pairs_test, selected_synsets, senses_per_sentence, report=False, verbose=False):
    dict_pt_verbs = {'tratar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'estabelecer_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'marcar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'vir_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'colocar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'fechar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'dar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'cair_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'encontrar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'registrar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'levar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'receber_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'apresentar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'passar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'deixar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'chegar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'ficar_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'fazer_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'ter_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0},\
            'ser_tag': {'total_in_ambiguous': 0, 'total_out_ambiguous': 0, 'hint': 0}}
        
    hint = 0
    total_prec = 0
    total_reca = 0

    for ix, instance in enumerate(test_dataset):
        sentence = pairs_test[ix][0].lower()
        senses = senses_per_sentence[ix]
        if len(senses) == 0:
            continue
        output_words = predictor.predict_instance(instance)['predicted_tokens']
        
        for pos, sense in senses:
            if len(output_words) > pos:  
                pred = output_words[pos]
                if pred in selected_synsets:
                    dict_pt_verbs[sentence.split()[pos]]['total_out_ambiguous'] += 1
                    total_prec += 1
                    if sense == pred:
                        dict_pt_verbs[sentence.split()[pos]]['hint'] += 1
                        hint += 1

            total_reca += 1
            dict_pt_verbs[sentence.split()[pos]]['total_in_ambiguous'] += 1
            
        if verbose:
            print('-O-')

    precision = (hint / total_prec) if hint else 0
    recall = (hint / total_reca) if hint else 0
    f1 = (2 * precision * recall / (precision + recall)) if hint else 0
    
    if report:
        return f1, precision, recall, dict_pt_verbs
    else:
        return f1

def main(name_file='all_f1', train_dir='all', test_dir='test', dir_files='data/disambiguation/', dir_results='results_2/', max_length=120, cuda_id=0, cuda=True, n_epochs=9, seed=0, lr=0.0001):
    
    dir_train = os.path.join(dir_files, train_dir)
    dir_test = os.path.join(dir_files, test_dir)
    dir_results = os.path.join(dir_results, train_dir, name_file)
    os.makedirs(dir_results, exist_ok=True)
    
    input_lang, output_lang, pairs_train, pairs_test, senses_per_sentence = prepare_data(name_file, 'verbs_selected_lemma', max_length=max_length, dir_train=dir_train, dir_test=dir_test)
    selected_synsets = np.load(os.path.join(dir_files, 'selected_synsets.npy'))

    reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=WordTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read(os.path.join(dir_train, name_file + '.tsv'))
    validation_dataset = reader.read(os.path.join(dir_test, 'verbs_selected_lemma.tsv'))

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)

    encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})
    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    max_decoding_steps = 100   # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8,
                          use_bleu=True).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=8,
                      cuda_device=cuda_id)

    trainer.train()

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      patience=7,
                      num_epochs=1,
                      cuda_device=cuda_id)

    best_metric = 0
    metrics = []
    precisions = []
    recalls = []
    for i in range(0, 30):
        print('Epoch: {}'.format(i))
        trainer.train()

        predictor = SimpleSeq2SeqPredictor(model, reader)
        if True:
            metric, precision, recall, report = evaluate_acc(predictor, validation_dataset, pairs_test, selected_synsets, senses_per_sentence, report=True, verbose=False)
            metrics.append(metric)
            precisions.append(precision)
            recalls.append(recall)
            if metric > best_metric:
                best_metric = metric
                res = get_stats(report, pairs_train, pairs_test)
                res.to_csv(f'{dir_results}/report_allen.csv')
                with open(os.path.join(dir_results, "allen.th"), 'wb') as f:
                    torch.save(model.state_dict(), f)
                print('-----best----', best_metric)

    np.save(os.path.join(dir_results, 'metrics_allen.npy'), metrics)
    np.save(os.path.join(dir_results, 'recalls_allen.npy'), recalls)
    np.save(os.path.join(dir_results, 'precisions_allen.npy'), precisions)

    #with open(os.path.join(dir_results, "allen.th"), 'rb') as f:
    #    model.load_state_dict(torch.load(f))
    #f1, precision, recall, report = evaluate_acc(predictor, validation_dataset, pairs_test, selected_synsets, senses_per_sentence, report=True, verbose=False)
    #print('f1 score:', f1, 'precision:', precision, 'recall:', recall)

    #res = get_stats(report, pairs_train, pairs_test)
    #res.to_csv(f'{dir_results}/report_allen.csv')
                                                                             
if __name__ == '__main__':
    fire.Fire(main)
