# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import codecs

import numpy as np
from numpy.lib.stride_tricks import as_strided
import copy
from scipy import fftpack

from hyperparams import Hyperparams as hp
import tensorflow as tf

import os
import librosa
import pickle
import copy
import numpy as np
from pprint import pprint
from scipy import signal, io
import re 
import json


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config

class AudioProcessor(object):
    def __init__(self,
                 bits=None,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 ref_level_db=None,
                 num_freq=None,
                 power=None,
                 preemphasis=None,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 **kwargs):

        print(" > Setting up Audio Processor...")

        self.bits = bits
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = 0 if mel_fmin is None else mel_fmin
        self.mel_fmax = mel_fmax
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        print(" | > Audio Processor attributes.")
        members = vars(self)
        for key, value in members.items():
            print("   | > {}:{}".format(key, value))

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spec):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spec))

    def _build_mel_basis(self, ):
        n_fft = (self.num_freq - 1) * 2
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _normalize(self, S):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        if self.signal_norm:
            S_norm = ((S - self.min_level_db) / - self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm :
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def _denormalize(self, S):
        """denormalize values"""
        S_denorm = S
        if self.signal_norm:
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm) 
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db /
                    self.max_norm) + self.min_level_db
                return S_denorm
        else:
            return S

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000.0 * self.sample_rate)
        print(" | > fft size: {}, hop length: {}, win length: {}".format(
            n_fft, hop_length, win_length))
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def melspectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)  # Convert back to linear
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        else:
            return self._griffin_lim(S**self.power)

    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        D = self._denormalize(mel_spectrogram)
        S = self._db_to_amp(D + self.ref_level_db)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        else:
            return self._griffin_lim(S**self.power)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.1 sec margin """
        margin = int(self.sample_rate * 0.1)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=40, frame_length=1024, hop_length=256)[0]

    # WaveRNN repo specific functions
    # def mulaw_encode(self, wav, qc):
    #     mu = qc - 1
    #     wav_abs = np.minimum(np.abs(wav), 1.0)
    #     magnitude = np.log(1 + mu * wav_abs) / np.log(1. + mu)
    #     signal = np.sign(wav) * magnitude
    #     # Quantize signal to the specified number of levels.
    #     signal = (signal + 1) / 2 * mu + 0.5
    #     return signal.astype(np.int32)

    # def mulaw_decode(self, wav, qc):
    #     """Recovers waveform from quantized values."""
    #     mu = qc - 1
    #     # Map values back to [-1, 1].
    #     casted = wav.astype(np.float32)
    #     signal = 2 * (casted / mu) - 1
    #     # Perform inverse of mu-law transformation.
    #     magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
    #     return np.sign(signal) * magnitude

    def load_wav(self, filename, encode=False):
        x, sr = librosa.load(filename, sr=self.sample_rate)
        if self.do_trim_silence:
            x = self.trim_silence(x)
        # sr, x = io.wavfile.read(filename)
        assert self.sample_rate == sr
        return x

    def encode_16bits(self, x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    def quantize(self, x):
        return (x + 1.) * (2**self.bits - 1) / 2

    def dequantize(self, x):
        return 2 * x / (2**self.bits - 1) - 1


def texts_to_phonemes(fpaths,texts,outputfile='texts-phoneme.csv',alphabet=False):
    from PETRUS.g2p.g2p import G2PTranscriber
    transcript = os.path.join(hp.data, outputfile)
    if alphabet == True:
        alphabet_list=[]
        alpha=os.path.join(hp.data, 'phoneme-alphabet.csv')

    transcript= codecs.open(transcript, 'w', 'utf-8')
    for i in range(len(texts)):
        words = texts[i].strip().lower().split(' ')
        transcrito = [] 
        for word in words:
            g2p = G2PTranscriber(word, algorithm='silva')
            transcription = g2p.transcriber()
            transcrito.append(transcription)
            if alphabet == True:
                for caracter in transcription:
                    if caracter not in alphabet_list:
                        alphabet_list.append(caracter)

        frase = str(fpaths[i])+'=='+"_".join(transcrito)+'\n'
        transcript.write(frase)
    if alphabet == True:
        alphabet = codecs.open(alpha, 'w', 'utf-8')
        for i in alphabet_list:
            alphabet.write(i)


def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
 
    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    # to amplitude
    mag = np.power(10.0, mag * 0.05)
    if hp.vocoder == 'griffin_lim':
        # wav reconstruction
        wav = griffin_lim(mag**hp.power)
    elif hp.vocoder == 'RTISI-LA' : # RTISI-LA
        wav = iterate_invert_spectrogram(mag**hp.power)
        
    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def iterate_invert_spectrogram(X_s, n_iter=10, verbose=False,
                               complex_input=False):
    """
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0 and not complex_input:
            X_t = invert_spectrogram(X_best)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(reg, np.abs(est))
        phase = phase[:len(X_s)]
        X_s = X_s[:len(phase)]
        X_best = X_s * phase
    X_t = invert_spectrogram(X_best)
    return np.real(X_t)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')

def guided_attention(g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((hp.max_N, hp.max_T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(hp.max_T) - n_pos / float(hp.max_N)) ** 2 / (2 * g * g))
    return W

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_spectrograms(fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    

    # Reduction
    re_mel = mel[::hp.r, :]
    return fname, re_mel, mel

