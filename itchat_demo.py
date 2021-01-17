import itchat
from itchat.content import *
import requests
import json, time

import evaluate, helper, config


from data_loader import DataLoader

#%%
class Robot:

    def __init__(self, model = "seq2seq"):
        self.model = model
        
        if self.model == "seq2seq":
            self.loader = DataLoader( 'data/english/' , False )
            self.encoder, self.decoder = helper.load( config.MODEL_PREFIX )

    def get_response(self, input_msg):
        if self.model == "seq2seq":
            result, _ = evaluate.evaluate( self.encoder, self.decoder, self.loader, input_msg )
            out_msg = " ".join( result[:-1] )
            
            print( out_msg)
            
            return  out_msg
    
        return "你好，我有事不在（自动回复）"

r = Robot('seq2seq')
#%%

@itchat.msg_register( [TEXT] )
def something(msg, isGroupChat = False):
    
    if msg.type != itchat.content.TEXT:
        print("Wrong! Not a text")
        
       
    f = open( "msg%f.json" % time.time(), "w", encoding = "utf-8" )
    
    json.dump( msg, f , ensure_ascii = False,  indent = 4 )
    
    f.close()
    
    
        
    x = msg.text
    print( "Message Got: %s" % x )

#    reply = r.get_response( x )
    
    reply = r.get_response(x )

    msg.user.send( reply )
    
    return


#@itchat.msg_register( [ATTACHMENT, MAP ] )
#def dosomethingelse( msg, isGroupChat = False) :
#    # do something
#    print("I am here") 
#    return

#@itchat.msg_register( [TEXT] )
#def reply_group_chat( msg, isGroupChat = True ):
#    if msg.isAt: # somebody else at you
#        # do something
#        pass
#    
#    if msg.actualNickName == "ABC":
#        # do soemthing
#        pass

if __name__ == "__main__":
    itchat.auto_login( hotReload = True )
    itchat.run( True )
    
    