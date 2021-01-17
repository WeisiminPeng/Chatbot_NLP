import yaml, json
import os, glob, random

import config, helper

#%%
#read .yml
class DataLoader:
    def __init__( self, folder_path, force_reload = False ):
        self.folder_path = folder_path    
        
        should_reload_data = force_reload
        should_reload_data |= not os.path.exists( "conversations.json") 
        should_reload_data |= not os.path.exists( "indexed_conversations.json") 
        should_reload_data |= not os.path.exists( "word2num.json" ) 
        should_reload_data |= not os.path.exists( "num2word.json" ) 
        
        if should_reload_data:
            print("Loading conversations from raw conversations into DataLoader")
            
            self.word2num = { '<PAD>':config.PAD_token, "<EOS>":config.EOS_token, "<SOS>":config.SOS_token }
            self.num2word = { config.PAD_token:'<PAD>', config.EOS_token:"<EOS>", config.SOS_token:"<SOS>" }
            self.conversations = []
            self.indexed_conversations = []
        
            self.reload_data()
            self.__write_dict()
        else:
            print("Loading conversations from existing files into DataLoader")
            self.conversations = json.load( open("conversations.json", "r", encoding = "utf-8") )
            self.indexed_conversations = json.load( open("indexed_conversations.json", "r", encoding = "utf-8") )
            self.word2num = json.load( open("word2num.json", "r", encoding = "utf-8") )
            self.num2word = json.load( open("num2word.json", "r", encoding = "utf-8") )
            
            
            self.word2num = { k:int(v) for k,v in self.word2num.items()}
            self.num2word = { int(k):v for k,v in self.num2word.items()}
   
    def reload_data( self ):
        """
            Load data from DATA_PATh
            
            and return a vector of pairs. 
            In each pair, the first element is number vector of "bad writing" from Google Translate
            the second element is number vector of "good writing" from NYT
        """
        files = glob.glob( os.path.join( self.folder_path , "*.yml" ) )
        
        self.conversations = []
        self.indexed_conversations = []

        for filename in files:
            self.load_yaml_file( filename )
                
    def load_yaml_file( self, filename ):
        
        try:
            data = yaml.load( open(filename, 'r', encoding = 'utf-8') )
            self.conversations += data['conversations']
            self.__process_data()
        except Exception as e:
            print(e)
            print("unable to load yaml file:", filename )
    
    def __process_data( self ): 
        """ 
        add data into word2num and num2word dictionary
        """
        for conversation in self.conversations:
            if type(conversation) is not list: continue # I don't care one-sentence conversation
            current_indexed_conversation = []
            for sentence in conversation:
                sentence = helper.normalizeString( sentence )
                
                sentence_arr = sentence.split()

                for word in sentence_arr:
                    if word in self.word2num: continue
                    idx = len( self.word2num )
                    self.word2num[word] = idx
                    self.num2word[idx] = word
                    
                    
                current_indexed_conversation.append( [ self.word2num[word] for word in sentence_arr ] )
                    
            self.indexed_conversations.append( current_indexed_conversation )
        
    def __write_dict( self ):
        """write conversations, word2num, and num2word into file"""
        json.dump( self.conversations, open( "conversations.json", "w", encoding = "utf-8" ) )
        json.dump( self.indexed_conversations, open( "indexed_conversations.json", "w", encoding = "utf-8" ) )
        json.dump( self.word2num, open( "word2num.json", "w", encoding = "utf-8" ) )
        json.dump( self.num2word, open( "num2word.json", "w", encoding = "utf-8" ) )
        
    
    def num_of_tokens( self ): #tell you how large the data size is
        """
        return how many unqiue words are in the corpus
        """
        return len( self.word2num )
    
    
    def sentence_to_num_array( self, sentence ):
        """
            Input: a string of sentence. e.g. "hello world"
            Output: a vector of number representing the sentence. e.g. [231, 320]
            
                    # This block is equivalent to last line
                    #    temp = sentence.split() # array of string
                    #    rst = []
                    #    for _ in temp:
                    #        if _ in WORD2NUM: rst.append( _ )
                    #        else: rst.append( -1 )
                    #    return rst
        """
        sentence = helper.normalizeString( sentence ) # remove non-ascii character
        return [ self.word2num[_] if _ in self.word2num else 0 for _ in sentence.split() ]

 
    def num_array_to_sentence( self, arr ):
        """
            Input: a string of sentence. e.g. "hello world"
            Output: a vector of number representing the sentence. e.g. [2, 3]
        """
        return " ".join( [ self.num2word[ _ ] if _ in self.num2word else '' for _ in arr ] )

        
    
    def get_a_random_conversatinon( self, use_index = True ):
        if use_index:
            conversation = random.choice( self.indexed_conversations )
        else:
            conversation = random.choice( self.conversations )
        start_idx = random.randrange(0, len( conversation ) - 1 )
        
        return ( conversation[start_idx:start_idx+2] )
        
    def get_random_conversations( self, num , use_index = True ):
        return [ self.get_a_random_conversatinon( use_index ) for _ in range(num) ]
    
    
    def get_all_pairs( self , use_index = False ):
        pairs = []
        if use_index: conversations = self.indexed_conversations
        else: conversations = self.conversations
        for conversation in conversations:
            for i in range( len(conversation) - 1):
                pairs.append( [conversation[i], conversation[i+1]] )
    
        print( "There are %d pairs in the dataset" % len(pairs))
        return pairs
    
if __name__ == "__main__":
    loader = DataLoader( 'data/english/' , True )  #position of data
    x = loader.indexed_conversations
    word2num = loader.word2num
    num2word = loader.num2word
    pair = loader.get_a_random_conversatinon()
    pairs = loader.get_all_pairs()