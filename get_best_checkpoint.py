# -*- coding: utf-8 -*-
# /usr/bin/python2

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
from shutil import copyfile

from matplotlib import pyplot as plt
from librosa import  display
def Text2Mel_calc_loss(Y,Y_logits,mels,alignments,gts):
    tf.reset_default_graph()

    with tf.Session() as sess:
        mels= tf.convert_to_tensor(mels)

        Y = tf.convert_to_tensor(Y)
        Y_logits= tf.convert_to_tensor(Y_logits)
        alignments = tf.convert_to_tensor(alignments)
        gts=tf.convert_to_tensor(gts)

        

        
        # mel L1 loss
        loss_mels = tf.reduce_mean(tf.abs(Y - mels))

        # mel binary divergence loss
        loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_logits, labels=mels))

        # guided_attention loss
        '''A = tf.pad(alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
        attention_masks = tf.to_float(tf.not_equal(A, -1))
        loss_att = tf.reduce_sum(tf.abs(A*gts) * attention_masks)
        mask_sum = tf.reduce_sum(attention_masks)
        loss_att /= mask_sum'''
 
        total_loss= loss_mels + loss_bd1#+loss_att
        sess.run(tf.global_variables_initializer())
        loss= sess.run(total_loss)
        # total loss
        return loss

def _load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel = "mels/{}".format(fname.replace("wav", "npy"))
    mags = "mags/{}".format(fname.replace("wav", "npy"))
    mels = np.load(mel)
    mags = np.load(mags)
    
    return mels, mags
                
def get_best_checkpoint(logdir_path=hp.logdir+ "-1"):

    # Load validation data 
    fpaths,L = load_data("validation")
    mels = []
    mags = []
    for fpath in fpaths:
        mel, mag = _load_spectrograms(fpath)
        zeros = np.zeros((hp.max_T, hp.n_mels), np.float32) 
        zeros[:len(mel)] = mel
        mel = zeros
        mels.append(mel)
        mags.append(mag)

    


    best_validation_loss = 999999999999999
    files= os.listdir(logdir_path)
    list_checkpoints= [i for i in files if i.endswith('.meta')]
    bestdir= os.path.join(logdir_path,'bestmodel')
    if not os.path.isdir(bestdir):
        os.mkdir(bestdir)
    for checkpoint in list_checkpoints:
        tf.reset_default_graph()
        # Load graph
        g = Graph(mode="synthesize"); print("Graph loaded")
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Restore parameters
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
            saver1 = tf.train.Saver(var_list=var_list)
            
            saver1.restore(sess, os.path.join(logdir_path,checkpoint.split('.meta')[0]))
            print("Text2Mel Restored!")

            # Feed Forward
            ## mel
            Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
            Y_logits = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
            prev_max_attentions = np.zeros((len(L),), np.int32)
            alignments =[]
            gts = []
            for j in tqdm(range(hp.max_T)):
                _gs, _Y, _Y_logits,_gts,_max_attentions, _alignments = \
                    sess.run([g.global_step, g.Y,g.gts,g.Y_logits, g.max_attentions, g.alignments],
                            {g.L: L,
                            g.mels: Y,
                            g.prev_max_attentions: prev_max_attentions})
                Y[:, j, :] = _Y[:, j, :]
                Y_logits[:, j, :]=Y_logits[:, j, :]
                prev_max_attentions = _max_attentions[:, j]
                alignments.append(_alignments)
                gts.append(_gts)
                
        total_loss = 0
        mels_len=len(mels)
        for i in range(mels_len):
                loss = Text2Mel_calc_loss(Y[i],Y_logits[i],mels[i],alignments[i],gts[i])
                
                total_loss += loss
        total_loss /= mels_len
        print('total loss:',total_loss,' best validation loss: ',best_validation_loss)
        if total_loss < best_validation_loss:
            checkpoint_name= checkpoint.split('.meta')[0]
            print('loss: ',loss,'best validation loss:',best_validation_loss, 'checkpoint:',checkpoint_name)
            best_checkpoint_files= [i for i in files if i.split('.')[0]== checkpoint_name]
            for check_file in best_checkpoint_files:
                copyfile(str(os.path.join(logdir_path,check_file)), os.path.join(logdir_path,'bestmodel','bestmodel.'+check_file.split('.')[1]))
            #os.system('cp ' +str(os.path.join(logdir_path,checkpoint.split('.meta')[0]))+'* ', os.path.join(logdir_path,'bestmodel'))
            best_validation_loss = total_loss
            best_checkpoint_saved = checkpoint_name
    print('the best checkpoint: ',best_checkpoint_saved,'validation loss:',best_validation_loss)

if __name__ == '__main__':
    get_best_checkpoint()
    print("Done")


