# -*- coding: utf-8 -*-

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata

def load_vocab():
    if hp.phoneme == True:
        char2idx = {char: idx for idx, char in enumerate(hp.phoneme_vocab)}
        idx2char = {idx: char for idx, char in enumerate(hp.phoneme_vocab)}
    else:   
        char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
        idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    if hp.language == 'pt':
        accents = ('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT') #portuguese
        chars = [c for c in unicodedata.normalize('NFD', text) if c not in accents]
        text = unicodedata.normalize('NFC', ''.join(chars))# Strip accent
        print(text)
    else:
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accent
    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()
    
    if mode=="train":
        if "LJ" in hp.data:
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'metadata.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines[10:]:
                fname, _, text = line.strip().split("|")

                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                fpaths.append(fpath)
                #print('Antes da normalizacao',text)
                text = text_normalize(text) + "E"  # E: EOS
                #print('Apos normalizacao',text)
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                #print('converte index',text)
                #print('final',np.array(text, np.int32).tostring())
                texts.append(np.array(text, np.int32).tostring())

            return fpaths, text_lengths, texts
        elif "Portuguese" in hp.data:
            if hp.phoneme == True:
                     # Parse
                fpaths, text_lengths, texts = [], [], []
                transcript = os.path.join(hp.data, 'texts-phoneme.csv')
                lines = codecs.open(transcript, 'r', 'utf-8').readlines()
                for line in lines[:3054]+lines[3074:]:
                    #print(line)
                    fname,text = line.strip().split("==")

                    fpath = os.path.join(hp.data, "wavs", fname.split("/")[1])
                    fpaths.append(fpath)
                    #print('Antes da normalizacao',text)
                    text = text + "E"  # E: EOS
                    #print('Apos normalizacao',text)
                    text = [char2idx[char] for char in text]
                    text_lengths.append(len(text))
                    #print('converte index',text)
                    #print('final',np.array(text, np.int32).tostring())
                    texts.append(np.array(text, np.int32).tostring())

                return fpaths, text_lengths, texts
            else:
                # Parse
                fpaths, text_lengths, texts = [], [], []
                transcript = os.path.join(hp.data, 'texts.csv')
                lines = codecs.open(transcript, 'r', 'utf-8').readlines()
                for line in lines[:3054]+lines[3074:]:
                    fname,text = line.strip().split("==")

                    fpath = os.path.join(hp.data, "wavs", fname.split('/')[1])
                    fpaths.append(fpath)
                    #print('Antes da normalizacao',text)
                    text = text_normalize(text) + "E"  # E: EOS
                    #print('Apos normalizacao',text)
                    text = [char2idx[char] for char in text]
                    text_lengths.append(len(text))
                    #print('converte index',text)
                    #print('final',np.array(text, np.int32).tostring())
                    texts.append(np.array(text, np.int32).tostring())

                return fpaths, text_lengths, texts
        else: # nick or kate
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'metadata.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text, is_inside_quotes, duration = line.strip().split("|")
                duration = float(duration)
                if duration > 10. : continue

                fpath = os.path.join(hp.data, fname)
                fpaths.append(fpath)

                text += "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

            return fpaths, text_lengths, texts

    elif mode=="prepo":
            if "LJ" in hp.data:
                # Parse
                fpaths, text_lengths, texts = [], [], []
                transcript = os.path.join(hp.data, 'metadata.csv')
                lines = codecs.open(transcript, 'r', 'utf-8').readlines()
                for line in lines:
                    fname, _, text = line.strip().split("|")

                    fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                    fpaths.append(fpath)
                    #print('Antes da normalizacao',text)
                    text = text_normalize(text) + "E"  # E: EOS
                    #print('Apos normalizacao',text)
                    text = [char2idx[char] for char in text]
                    text_lengths.append(len(text))
                    #print('converte index',text)
                    #print('final',np.array(text, np.int32).tostring())
                    texts.append(np.array(text, np.int32).tostring())

                return fpaths, texts

            elif "Portuguese" in hp.data:
                # Parse
                fpaths, text_lengths, texts = [], [], []
                transcript = os.path.join(hp.data, 'texts.csv')
                lines = codecs.open(transcript, 'r', 'utf-8').readlines()
                for line in lines:
                    fname,text = line.strip().split("==")

                    fpath = os.path.join(hp.data, "wavs", fname.split('/')[1])
                    fpaths.append(fpath)
                    #print('Antes da normalizacao',text)
                    text = text_normalize(text)
                    #print('Apos normalizacao',text)
                    #text = [char2idx[char] for char in text]
                    #print('converte index',text)
                    #print('final',np.array(text, np.int32).tostring())
                    texts.append(text)

                return fpaths, texts
    elif mode=='validation':
        if "LJ" in hp.data:
            # Parse
            fpaths, text_lengths, texts,sents = [], [], [],[]
            transcript = os.path.join(hp.data, 'metadata.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines[0:10]:
                fname, _, text = line.strip().split("|")

                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                fpaths.append(fpath)
                #print('Antes da normalizacao',text)
                #print('Antes da normalizacao',text)
                sent = text_normalize(text) + "E"  # E: EOS
                sents.append(sent)
                    
                texts = np.zeros((len(sents), hp.max_N), np.int32)
                for i, sent in enumerate(sents):
                    texts[i, :len(sent)] = [char2idx[char] for char in sent]

            return fpaths, texts

        elif "Portuguese" in hp.data:
            if hp.phoneme == True:
                # Parse
                fpaths, text_lengths, texts,sents = [], [], [],[]
                transcript = os.path.join(hp.data, 'texts-phoneme.csv')
                lines = codecs.open(transcript, 'r', 'utf-8').readlines()
                for line in lines[3054:3074]:# fonetic balance phrases
                    #print(line)
                    fname,text = line.strip().split("==")

                    fpath = os.path.join(hp.data, "wavs", fname.split("/")[1])
                    fpaths.append(fpath)
                    #print('Antes da normalizacao',text)
                    #print('Antes da normalizacao',text)
                    sent = text_normalize(text).replace(',',' , ').replace('?',' ? ') + "E"  # E: EOS
                    sents.append(sent)
                    
                texts = np.zeros((len(sents), hp.max_N), np.int32)
                for i, sent in enumerate(sents):
                    texts[i, :len(sent)] = [char2idx[char] for char in sent]

                return fpaths, texts
            else:
                # Parse
                fpaths, text_lengths, texts,sents = [], [], [],[]
                transcript = os.path.join(hp.data, 'texts.csv')
                lines = codecs.open(transcript, 'r', 'utf-8').readlines()
             
                for line in lines[3054:3074]:
                    fname,text = line.strip().split("==")

                    fpath = os.path.join(hp.data, "wavs", fname.split('/')[1])
                    fpaths.append(fpath)
                    #print('Antes da normalizacao',text)
                    sent = text_normalize(text) + "E"  # E: EOS
                    sents.append(sent)
                    
                texts = np.zeros((len(sents), hp.max_N), np.int32)
                for i, sent in enumerate(sents):
                    texts[i, :len(sent)] = [char2idx[char] for char in sent]

                return fpaths, texts


    else: # synthesize on unseen test text.
        # Parse
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        if "Portuguese" in hp.data and hp.phoneme == True:
            sents = [line.split(" ", 1)[-1].strip() + "E" for line in lines]
            print('sents:',sents)
            for line in lines:
                print('split:',line.split(" ", 1)[-1])
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent]
            return texts
        else:
            sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent]
            return texts
        

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                fname = fname.decode("utf8")
                mel = "mels/{}".format(fname.replace("wav", "npy"))
                mags = "mags/{}".format(fname.replace("wav", "npy"))
                mels = np.load(mel)
                mags = np.load(mags)
                return fname, mels, mags

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.B,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=hp.B*4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch

