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

from matplotlib import pyplot as plt
from librosa import  display



frases=["A inauguração da vila é quarta ou quinta-feira",
        "Vote se você tiver o título de eleitor",
        "Hoje é fundamental encontrar a razão da existência humana",
        "A temperatura é mais amena à noite",
        "Em muitas cidades a população está diminuindo.",
        "Nunca se deve ficar em cima do morro",
        "Para as pessoas estranhas o panorama é desolador",
        "É bom te ver colhendo flores menino",
        "Eu finjo me banhar num lago ao amanhecer",
        "Sua sensibilidade mostrará o caminho",
        "A Amazônia é a reserva ecológica do globo",
        "O ministério mudou demais com a eleição",
        "Novas metas surgem na informática",
        "O capital de uma empresa depende de sua produção",
        "Se não fosse ela tudo teria sido melhor",
        "A principal personagem no filme é uma gueixa",
        "Espere seu amigo em casa",
        "A juventude tinha que revolucionar a escola",
        " A cantora terá quatro meses para ensaiar seu canto",
        "Esse tema foi falado no congresso."]


import os
from matplotlib import pylab as plt

def plot_alignment_with_text(alignment,text, info=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation=None)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel,fontsize = 'x-large')
    plt.ylabel('Encoder timestep',fontsize = 'x-large')
    #plt.yticks(range(len(text)), list(text.upper()),fontsize ='xx-large' )
    ax.set_yticks(range(len(text)))
    ax.set_yticklabels(list(text.upper()),fontsize ='medium')
    plt.tight_layout()
    return fig


def synthesize():
    # Load data
    L = load_data("synthesize")
    
    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]


        #plot aligments
        for i,al in enumerate(alignments):
            fig = plot_alignment_with_text(al,frases[i])
            fig.savefig(os.path.join(hp.sampledir,'align_'+str(i+1)+'_val.png'))

        for i, mag in enumerate(Y):
            print("Teste Working on file", i+1)
            
        
        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            wav = spectrogram2wav(mag)
            write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)
            np.save( hp.sampledir + "/{}.png".format(i+1),mag )#save mag 
            # transpose
            mag = mag.T

            # de-noramlize
            mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
            # to amplitude
            mag = np.power(10.0, mag * 0.05)
            #save spectrogram stft image
            mag = mag**hp.power
            display.specshow(librosa.amplitude_to_db(mag,ref=np.max), y_axis='log', x_axis='time')
            plt.title('Espectrograma STFT')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            #plt.savefig(hp.sampledir + "/{}.png".format(i+1))
            plt.cla()   # Clear axis
            plt.clf()


if __name__ == '__main__':
    synthesize()
    print("Done")


