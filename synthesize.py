# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm

from matplotlib import pyplot as plt
from librosa import  display

def synthesize():
    # Load data
    Y = load_data("test")
    
    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mel2World') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-3"))
        print("Network Restored!")

        for i, mag in enumerate(Y):
            print("Teste Working on file", i+1)
            
        
        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, world_tensor in enumerate(Z):
            print("Working on file", i+1)
            lf0,mgc,bap = tensor_to_world_features(world_tensor)
            wav = world2wav(lf0, mgc, bap)
            sf.write(hp.sampledir + "/{}.wav".format(i+1), wav,hp.sr_dataset)


if __name__ == '__main__':
    synthesize()
    print("Done")


