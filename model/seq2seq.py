import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

class Seq2seq(nn.Module):
    def __init__(self, input_lang, output_lang, encoder, decoder, tf_ratio, USE_CUDA=False):
        super(Seq2seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.tf_ratio = tf_ratio
        
        self.input_lang = input_lang
        self.output_lang = output_lang
        
        self.USE_CUDA = USE_CUDA

    def forward(self, input_batches, target_batches=[], train=False):
        
        batch_size = input_batches.size()[1]
        encoder_hidden = self.encoder.init_hidden(batch_size)

        encoder_outputs, encoder_hidden = self.encoder(input_batches, encoder_hidden)
        decoder_input = torch.LongTensor([self.input_lang.vocab.stoi['<sos>']] * batch_size)
    
        decoder_hidden = encoder_hidden
        decoder_context = torch.zeros(batch_size, self.decoder.hidden_size)
    
        all_decoder_outputs = torch.zeros(target_batches.data.size()[0], batch_size, len(self.output_lang.vocab.itos))

        if self.USE_CUDA:
            all_decoder_outputs = all_decoder_outputs.cuda()
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
    
        if train:
            use_teacher_forcing = np.random.random() < self.tf_ratio
        else:
            use_teacher_forcing = False
    
        if use_teacher_forcing:        
            # Use targets as inputs
            for di in range(target_batches.shape[0]):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input.unsqueeze(0), decoder_context, decoder_hidden, encoder_outputs)
            
                all_decoder_outputs[di] = decoder_output
                decoder_input = target_batches[di]
        else:        
            # Use decoder output as inputs
            for di in range(target_batches.shape[0]):            
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input.unsqueeze(0), decoder_context, decoder_hidden, encoder_outputs) 
            
                all_decoder_outputs[di] = decoder_output
            
                # Greedy approach, take the word with highest probability
                topv, topi = decoder_output.data.topk(1)            
                decoder_input = torch.LongTensor(topi.cpu()).squeeze()
                if self.USE_CUDA: decoder_input = decoder_input.cuda()
        
        if len(target_batches):
            return all_decoder_outputs, target_batches 
        else:
            return all_decoder_outputs