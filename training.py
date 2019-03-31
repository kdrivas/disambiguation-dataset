# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import pandas as pd
import time
import fire
import random
import os

from model.encoder import Encoder
from model.decoder import Decoder_luong
from model.seq2seq import Seq2seq

from src.utils import time_since
from src.utils import get_stats
from src.data import prepare_data
from src.data_loader import get_loader
from src.evaluator import evaluate_acc
from src.parallel import DataParallelCriterion, DataParallelModel

def train(input_var, target_var, model,  model_optimzier, clip, output_size, device, train=True):
    
    if train:
        model_optimzier.zero_grad()

    all_decoder_outputs, target_var = model(input_var, target_var, device, train)
    loss = nn.NLLLoss()(all_decoder_outputs.view(-1, output_size), target_var.contiguous().view(-1))          
    
    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        model_optimzier.step()
    
    return loss.item() 

def main(name_file, train_dir='all', test_dir='test', dir_files='data/disambiguation/', dir_results='results/', max_length=120, cuda_ids = [0, 1], cuda=True, n_epochs=13, seed=0):
    
    dir_train = os.path.join(dir_files, train_dir)
    dir_test = os.path.join(dir_files, test_dir)
    dir_results = os.path.join(dir_results, train_dir, name_file)
    os.makedirs(dir_results, exist_ok=True)
    
    attn_model = 'general'
    hidden_size = 712
    emb_size = 400
    n_layers = 2
    seed = 12
    dropout_p = 0.2
    tf_ratio = 0.5
    clip = 5.0

    batch_size = 50
    plot_every = 5
    start_eval = 5
    print_every = 50
    validate_loss_every = 100
    evaluate_every = 50

    train_losses = []
    validation_losses = []
    validation_acc = []
    best_metric = 0
    print_loss_total = 0
    plot_loss_total = 0

    if cuda: torch.cuda.set_device(cuda_ids[0])
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_lang, output_lang, pairs_train, pairs_test, senses_per_sentence = prepare_data(name_file, 'verbs_selected_lemma', max_length=max_length, dir_train=dir_train, dir_test=dir_test)
    selected_synsets = np.load(os.path.join(dir_files, 'selected_synsets.npy'))
    print(pairs_train[-1])
    encoder = Encoder(len(input_lang.vocab.stoi), hidden_size, emb_size, n_layers, dropout_p, USE_CUDA=cuda)
    decoder = Decoder_luong(attn_model, hidden_size, len(output_lang.vocab.stoi), emb_size, 2 * n_layers, dropout_p, USE_CUDA=cuda)
    model = Seq2seq(input_lang, output_lang, encoder, decoder, tf_ratio, cuda)
    device  = torch.device(cuda_ids[0] if cuda else 'cpu')
    if cuda:
        model = nn.DataParallel(model, device_ids=cuda_ids).cuda()

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()
    #if cuda:
    #    criterion = DataParallelCriterion(criterion, device_ids=cuda_ids).cuda()

    train_loader = get_loader(pairs_train, input_lang.vocab.stoi, output_lang.vocab.stoi, batch_size=batch_size)
    start = time.time()

    for epoch in range(1, n_epochs + 1): 
        # Shuffle data
        id_aux = np.random.permutation(np.arange(len(pairs_train)))
        pairs_train = pairs_train[id_aux]

        model.train()
        print_loss_total = 0
        # Get the batches for this epoch

        for batch_ix, (input_var, _, target_var, _) in enumerate(train_loader):
            # Transfer to GPU
            input_var, target_var = input_var.cuda(), target_var.cuda() # input_var.to(device), target_var.to(device)

            # Run the train function
            loss = train(input_var, target_var, model, model_optimizer, clip, decoder.output_size, device, train=train)
            torch.cuda.empty_cache()

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if batch_ix == 0 and epoch == 1: continue

            if batch_ix % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, 100 * batch_ix / round(len(pairs_train) / batch_size), print_loss_avg)
                train_losses.append(loss)
                print(print_summary)

            if epoch >= 4 and batch_ix % evaluate_every == 0:
                model.eval()

                metric = evaluate_acc(encoder, decoder, input_lang, output_lang, pairs_test, selected_synsets, senses_per_sentence, k_beams=1, report=False, max_length=max_length, cuda=cuda)
                if metric >= best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), f'{dir_results}/seq2seq.pkl')
                    torch.save(encoder.state_dict(), f'{dir_results}/enc.pkl')
                    torch.save(decoder.state_dict(), f'{dir_results}/dec.pkl')
                    print('Saving weights')
                validation_acc.append(metric)
                print(f'Validate metric: {metric}')

                model.train()
                
                
    np.save(f'{dir_results}/train_losses.npy', train_losses)
    np.save(f'{dir_results}/validation_losses.npy', validation_losses)
    np.save(f'{dir_results}/validation_acc', validation_acc)

    model.load_state_dict(torch.load(f'{dir_results}/seq2seq.pkl'))
    encoder.eval()
    decoder.eval()
    f1, precision, recall, report = evaluate_acc(encoder, decoder, input_lang, output_lang, pairs_test, selected_synsets, senses_per_sentence, k_beams=1, report=True, max_length=max_length, cuda=cuda)
    print('f1 score:', f1, 'precision:', precision, 'recall:', recall)

    res = get_stats(report, pairs_train, pairs_test)
    res.to_csv(f'{dir_results}/report.csv')
    
if __name__ == '__main__':
    fire.Fire(main)

