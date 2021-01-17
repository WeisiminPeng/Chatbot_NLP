import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from data_loader import DataLoader

import config, helper



def evaluate(encoder, decoder, loader, sentence ):
    """
    Inputs:
        encoder and decoder are PyTorch nn objects
        loader is the data_loader.DataLoader object
        sentence is a string of sentence, e.g. "How are you?"
    """
    max_length = config.MAX_SENTENCE_LEN
    sentence = helper.normalizeString( sentence )
    
    input_variable = helper.variableFromArray( loader.sentence_to_num_array(sentence) )
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if config.USE_CUDA else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[config.SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if config.USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == config.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append( loader.num2word[ni] )

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if config.USE_CUDA else decoder_input

    return decoded_words, decoder_attentions[:di + 1]



def evaluateRandomly(encoder, decoder, loader, n=1):
    for i in range(n):
        try:
            pair = loader.get_a_random_conversatinon( use_index = False)
            print('Human:', pair[0])
#            print('=', pair[1])
            output_words, attentions = evaluate(encoder, decoder, loader, pair[0])
            output_sentence = ' '.join(output_words)
            print('Bot:', output_sentence)
            print('')
        except Exception as e:
            print( e )
            print( "error evaluating sentence:", pair )

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence, encoder, decoder , loader):
    output_words, attentions = evaluate(
        encoder, decoder, loader, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)



if __name__ == "__main__":
    loader = DataLoader( 'data/english/' , False )
        
    if helper.model_exist( config.MODEL_PREFIX ):
        # if .pt model exist, then load the model directly 
        encoder, decoder  = helper.load(  config.MODEL_PREFIX )
        
        if config.USE_CUDA:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            
        evaluateRandomly(encoder, decoder, loader, n = 10 )
#        evaluateAndShowAttention( 'hi, 你会说中文吗?'  , encoder, decoder, loader)
        evaluateAndShowAttention( "What can you do in China?"  , encoder, decoder, loader)
        evaluateAndShowAttention( "You are jealous"  , encoder, decoder, loader)
        
        
        #%%
#        while True:
#            line = input("Human: ")
#            if len( line ) == 0: break
#            evaluateAndShowAttention( line  , encoder, decoder, loader)
        #%%
        
    else:
        print( "Model not found. Bye!" )
        


