# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
import tqdm
from hyperparams import Hyperparams as hp
import codecs

from utils import AudioProcessor,load_config
import re
import phonemizer
from phonemizer.phonemize import phonemize

# Regular expression matchinf punctuations, ignoring empty space
pat = r'['+hp.phoneme_punctuations[:-1]+']+'

import re

def text2phone(text, language):
    '''
    Convert graphemes to phonemes.
    '''
    seperator = phonemizer.separator.Separator(' |', '', '|')
    #try:
    punctuations = re.findall(pat, text)
    ph = phonemize(text, separator=seperator, strip=False, njobs=1, backend='espeak', language=language)
    # Replace \n with matching punctuations.
    ph = ph[:-1].strip() # skip the last empty character
    # Replace \n with matching punctuations.
    if punctuations:
        # if text ends with a punctuation.
        if text[-1] == punctuations[-1]:
            for punct in punctuations[:-1]:
                ph = ph.replace('| |\n', '|'+punct+'| |', 1)
            try:
                ph = ph + punctuations[-1]
            except:
                print(text)
        else:
            for punct in punctuations:
                ph = ph.replace('| |\n', '|'+punct+'| |', 1)
    return ph

def phrase_to_phoneme(clean_text, language):
    phonemes = text2phone(clean_text, language)
#    print(phonemes.replace('|', ''))
    if phonemes is None:
        print("!! After phoneme conversion the result is None. -- {} ".format(clean_text))
    
    lista = phonemes.split('||')
    texto = [x.replace('|','') for x in lista]
    return ' '.join(texto)


def texts_to_phonemes(fpaths,texts):
    transcript = os.path.join(hp.data, 'texts-phoneme.csv')
    transcript= codecs.open(transcript, 'w', 'utf-8')
    #print('Texts:',texts)
    for i in range(len(texts)):
        transcrito = phrase_to_phoneme(texts[i],hp.language)
        frase = str(fpaths[i].replace(hp.data,''))+'=='+transcrito+'\n'
        transcript.write(frase)
    
    
    
if __name__ == "__main__":
        
    # Load data
    fpaths, texts = load_data(mode="prepo") # list

    c = load_config('config.json')

    ap = AudioProcessor(**c.audio)


    if hp.phoneme == True:
        if hp.language =='pt-br':
            texts_to_phonemes(fpaths,texts)

    '''for fpath in tqdm.tqdm(fpaths):
        x = ap.load_wav(fpath, ap.sample_rate)
        fname = os.path.basename(fpath)
        mel = ap.melspectrogram(x.astype('float32')).astype('float32')
        # Marginal padding for reduction shape sync.
        num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
        mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
        
        # Reduction
        re_mel = mel[::hp.r, :]

        if not os.path.exists("mels"): os.mkdir("mels")
        if not os.path.exists("mags"): os.mkdir("mags")

        np.save("mels/{}".format(fname.replace("wav", "npy")), re_mel)
        np.save("mags/{}".format(fname.replace("wav", "npy")), mel)'''
