from hyperparams import Hyperparams as hp

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from data_load import text_normalize,load_vocab
#from utils import *
from scipy.io.wavfile import write
from tqdm import tqdm
from librosa import  display
from IPython.display import Audio
import os

# Load graph
g = Graph(mode="synthesize"); print("Graph loaded")

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# Restore parameters
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
saver1 = tf.train.Saver(var_list=var_list)
saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
print("Text2Mel Restored!")

var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
saver2 = tf.train.Saver(var_list=var_list)
saver2.restore(sess, os.path.join(tf.train.latest_checkpoint(hp.logdir + "-2"))
print("SSRN Restored!")

transcript = os.path.join(hp.data, 'texts.csv')
lines = codecs.open(transcript, 'r', 'utf-8').readlines()
os.makedirs('mels', exist_ok=True)
os.makedirs('mels_test', exist_ok=True) 
char2idx, idx2char = load_vocab()
               
for i in range(len(lines)):
        line = lines[i]
        print(line)
        out = 'mels'
        fname,text = line.strip().split("==")
        file_id = '{:d}'.format(i).zfill(5)
        file_name = os.path.basename(fname)
        print(file_name)
        if int(file_name.split('-')[1].replace('.wav','')) >= 5655 and int(file_name.split('-')[1].replace('.wav',''))<=5674:
            out='mels_test'

        frase = '1 '+text
        #normalize remove inavalid characters
        frase = text_normalize(frase.split(" ", 1)[-1]).strip() + "E" # text normalization, E: EOS

        print('normalized text:',frase)
      

      
        #convert characters to numbers
        text = np.zeros((1, hp.max_N), np.int32)#hp.max_N = 128, is the max number for characters 
        text[0, :len(frase)] = [char2idx[char] for char in frase]

        print('converted text:',text)

        L = text
        # Feed Forward
        ## mel
        # note: hp.max_T can be changed depending on the phrase to be synthesized, the default value is 210, which generates an audio of maximum 10 seconds, if it decreases this value can obtain a greater speed of synthesis.
        hp.max_T = 210 
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = sess.run([g.global_step, g.Y, g.max_attentions, g.alignments], {g.L: L,g.mels: Y, g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

            

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        for i, mel in enumerate(Z):
            print("Working on file", i+1)
            #wav = spectrogram2wav(mag)
            mel = mel.T
            np.save(os.path.join(out,file_id+".npy"), mel)




