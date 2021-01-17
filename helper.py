import time, math, re, os, json, glob, yaml
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


import config

#%% Show time
# timeing training model
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#for debug
# tell you how much time you used and the loss value
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


#%%
# Using chart to show the lost after training    
#learning rate
def showPlot(points):
    print( 'showPlot with parameter:', points )
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
    
#%% transfer unique to Ascii  
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

#s=>sentence 
def normalizeString(s, language = "English" ):
    s = str( s ) # sanity check
    if language == "English":
        # Lowercase, trim, and remove non-letter characters
        
        try:
            s = unicodeToAscii(s.lower().strip())#Remove the spaces and convert all to lowercase
            s = re.sub(r"([.!?])", r" \1", s)#Turn punctuation marks into spaces plus punctuation marks
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)#Unconventional symbols are all converted to spaces
        except Exception as e:
            print( e )
            print( "Unable to normalize string:", s)

    elif language == "Chinese": 
        # TODO: use the function we created for Poet_Generator
        pass
    
    return s.strip().rstrip()

#%%
# change array of number to torch variable of the array
def variableFromArray( indexes ):
    """
    Input: an array of number
    Output: torch variable of the array
    """
    indexes.append( config.EOS_token )
    
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    
    if config.USE_CUDA:
        return result.cuda()
    else:
        return result


def variablesFromPair( one_pair ):
    """
        Input: a pair of vector. e.g.   [  [202, 323, 100], [5, 10, 32, 91]   ]
        Output: a pair of corresponding pyTorch array
    """
    input_variable = variableFromArray( one_pair[0] )
    target_variable = variableFromArray( one_pair[1] )
    
    return (input_variable, target_variable)    
    

#%%

 


#%%
#save result    
def model_exist( prefix_name ):
    return os.path.exists(prefix_name + ".pt")
     

def save( encoder, decoder, prefix_name ):
    torch.save( [encoder, decoder ], prefix_name + ".pt" )
    print('Model Saved')


def load( prefix_name ):
    [encoder, decoder ] = torch.load( prefix_name + ".pt"  )
    print("Model Loaded")
    return encoder, decoder

    