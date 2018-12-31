# -*- coding: utf-8 -*-
class Hyperparams:
    '''Hyper parameters'''
    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
    
    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Model
    r = 4 # Reduction factor. Do not change this.
    dropout_rate = 0.05
    vocoder = 'RTISI-LA' # or 
    #vocoder = 'griffin_lim'
    e = 128 # == embedding
    d = 256 # == hidden units of Text2Mel
    c = 512 # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = "/home/edresson/Projetos-PTI/TCC/text-dataset/App/Base/TTS-Portuguese/"
    # data = "/data/private/voice/kate"
    language = 'pt' # or 'eng'
    phoneme = True
    if phoneme == False and language == 'pt':
        test_data = 'phonetically-balanced-sentences.txt'
    elif phoneme == True and language == 'pt':
        test_data = 'phonetically-balanced-sentences-phoneme.txt'
    else:
        test_data = 'harvard_setences.txt'

    #vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding, E: EOS. #english
    vocab = "PE abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû"#abcdefghijklmnopqrstuvwxyzçãõâôêíîúûñáéó.?" # P: Padding, E: EOS. #portuguese
    #portugues falta acento no a :"abcdefghijklmnopqrstuvwxyzçãõâôêíîúûñáéó.?"
    phoneme_vocab = "ˈoʧi.tulʊʤɪpaʒnsdk̃eɾvmzgɐ͂ɛxfbɣ,_ɔXqɲʃʎĩẽõhũŋcrɳ E"
    max_N = 180 # Maximum number of characters. default:180
    max_T = 210 # Maximum number of mel frames. default:210

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "../savedir/logdir/LJ01"
    sampledir = '../savedir/samples'
    B = 32 # batch size
    num_iterations =10# 2000000
