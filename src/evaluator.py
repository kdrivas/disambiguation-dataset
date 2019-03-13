import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import re 
import numpy as np
from tqdm import tqdm, tqdm_notebook
from src.data import variable_from_sentence

class Beam():
    def __init__(self, decoder_input, decoder_context, decoder_hidden,
                    decoded_words=[], decoder_attentions=[], sequence_log_probs=[], decoded_index=[], decoder_arr_hidden=[], decoded_arr_output=[]):
        self.decoded_words = decoded_words
        self.decoded_index = decoded_index
        self.decoded_arr_output = decoded_arr_output
        self.decoder_attentions = decoder_attentions
        self.sequence_log_probs = sequence_log_probs
        self.decoder_input = decoder_input
        self.decoder_context = decoder_context
        self.decoder_hidden = decoder_hidden
        self.decoder_arr_hidden = decoder_arr_hidden

def evaluate(encoder, decoder, input_lang, output_lang, sentence, k_beams, USE_CUDA=True):
    with torch.no_grad():
        input_variable = variable_from_sentence(input_lang, sentence, USE_CUDA)
        input_length = input_variable.shape[0]

        encoder_hidden = encoder.init_hidden(1)
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[input_lang.vocab.stoi['<sos>']]]))
        decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
        decoder_hidden = encoder_hidden

        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        beams = [Beam(decoder_input, decoder_context, decoder_hidden)]
        top_beams = []

        # Use decoder output as inputs
        for di in range(MAX_LENGTH):      
            new_beams = []
            for beam in beams:
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                    beam.decoder_input, beam.decoder_context, beam.decoder_hidden, encoder_outputs)     

                # Beam search, take the top k with highest probability
                topv, topi = decoder_output.data.topk(k_beams)

                for ni, vi in zip(topi[0], topv[0]):
                    new_beam = Beam(None, decoder_context, decoder_hidden, 
                                        beam.decoded_words[:], beam.decoder_attentions[:], beam.sequence_log_probs[:])
                    new_beam.decoder_attentions.append(decoder_attention.squeeze().cpu().data)
                    new_beam.sequence_log_probs.append(vi)

                    if ni == input_lang.vocab.stoi['<eos>'] or ni == output_lang.vocab.stoi['<pad>']: 
                        new_beam.decoded_words.append('<EOS>')
                        top_beams.append(new_beam)

                    else:
                        new_beam.decoded_words.append(output_lang.vocab.itos[ni])                        

                        decoder_input = Variable(torch.LongTensor([[ni]]))
                        if USE_CUDA: decoder_input = decoder_input.cuda()
                        new_beam.decoder_input = decoder_input                        
                        new_beams.append(new_beam)                    

            new_beams = {beam: np.mean(beam.sequence_log_probs) for beam in new_beams}
            beams = sorted(new_beams, key=new_beams.get, reverse=True)[:k_beams]

            if len(beams) == 0:
                break

        if len(top_beams):
            top_beams = {beam: np.mean(beam.sequence_log_probs) for beam in top_beams}
        else:
            top_beams = {beam: np.mean(beam.sequence_log_probs) for beam in beams}

        # for beam in top_beams:
        #     print(beam.decoded_words, top_beams[beam])

        top_beams = sorted(top_beams, key=top_beams.get, reverse=True)[:k_beams]        

        decoded_words = top_beams[0].decoded_words
        for di, decoder_attention in enumerate(top_beams[0].decoder_attentions):
            decoder_attentions[di,:decoder_attention.size(0)] += decoder_attention

    return decoded_words, decoder_attentions[:len(top_beams[0].decoder_attentions)+1, :len(encoder_outputs)]

def evaluate_acc(encoder, decoder, input_lang, output_lang, pairs, senses_per_sentence, k_beams=3, verbose=False, report=False):
    
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
    total_senses = 0
    selected_synsets = np.load(Path.cwd() / 'data/disambiguation/selected_synsets.npy')
    for ix, (pair, senses) in enumerate(zip(pairs, senses_per_sentence)):
        sentence = pair[0].lower()
        if len(senses) == 0:
            continue
        output_words, attentions = evaluate(sentence, k_beams)
        torch.cuda.empty_cache()

        for pos, sense in senses:
                
            pred = output_words[attentions[:, pos].max(0)[1].item()]
            if pred in selected_synsets:
                dict_pt_verbs[sentence.split()[pos]]['total_out_ambiguous'] += 1

                total_senses += 1
                if sense == pred:
                    dict_pt_verbs[sentence.split()[pos]]['hint'] += 1
                    hint += 1
            dict_pt_verbs[sentence.split()[pos]]['total_in_ambiguous'] += 1
                            
        if verbose:
            print("----- ID")
            print(ix)
            print("----- tokens input")
            print(pair[0].lower())
            print("----- tokens real")
            print(pair[1].lower())
            print("----- output predecido")
            print(' '.join(output_words))
            print()
            print("--- hints:  {}   --- acc: {}".format(hint, (hint / total_senses if total_senses else 1)))
            
    if report:
        return hint / total_senses, dict_pt_verbs
    else:
        return hint / total_senses
        
def evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs):
    pair = random.choice(pairs)
    
    output_words, decoder_attn = evaluate(pair[0], 1)
    output_sentence = ' '.join(output_words)
    
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')
    
