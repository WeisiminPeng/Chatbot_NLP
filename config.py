import os, re
import torch
from torch import optim


#%% Global Setting
MAX_SENTENCE_LEN = 20
#set bucket  input=output=20

USE_CUDA =  torch.cuda.is_available()
#if you have GPU, choosetrue 

DATA_PATH = "data/"

if not os.path.exists( DATA_PATH ): os.mkdir( DATA_PATH )


#%%
CELL_TYPE = "gru" # TODO: torch.nn.gru or torch.nn.lstm
OPTIMIZER = optim.Adam # optim.SGD

# Cast the parameter into string, and combine them to MODEL_PREFIX
# MODEL_PREFIX is the filename we will use to save the model
str_optimizer = re.split( '[\.>\']', str( OPTIMIZER ) )
str_optimizer = list( filter( lambda x: len(x) , str_optimizer ) )
MODEL_PREFIX = os.path.join( DATA_PATH, "model_%s_%s"  % (CELL_TYPE, str_optimizer[-1]))
#dave model of pytorch
#%%
PAD_token = 0
SOS_token = 1
EOS_token = 2

