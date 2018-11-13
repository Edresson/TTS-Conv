
from matplotlib import pyplot as plt
import numpy as np
import librosa
from librosa import  display
import os
from utils import load_spectrograms
from hyperparams import Hyperparams as hp
# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2



cinco_org= np.load('../savedir/synthesized-audios/RTISI-LA/5-org.npy')

savedir='../savedir/spectogramas/'
vocoders = ['RTISI-LA','Griff-Lim']
for voco in vocoders:
    diretory = os.path.join('../savedir/synthesized-audios/',voco)
    arquivos = ['1.wav','5.wav','5-org.wav','3.wav','3-org.wav']
    for i in arquivos:
        _,_,mag= load_spectrograms(os.path.join(diretory,i))
        print  (os.path.join(diretory,i),' :',mag.shape)
        if i == '5.wav':
            mag = mag[:cinco_org.shape[0]][:]
        np.save( os.path.join(diretory,i.replace('.wav','')), mag)
        # transpose
        mag = mag.T

        # de-noramlize
        mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
        # to amplitude
        mag = np.power(10.0, mag * 0.05)
        mag = mag**hp.power
        display.specshow(librosa.amplitude_to_db(mag,ref=np.max), y_axis='log', x_axis='time')
        plt.title('Espectrograma STFT')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(os.path.join(savedir,voco,'spectrograma-'+voco+'-'+i.replace('.wav','')+'.png'))
        plt.cla()   # Clear axis
        plt.clf()
        #plt.show()

## R2 Calcule
import numpy as np
from sklearn.metrics import r2_score

y_true= np.load('../savedir/synthesized-audios/RTISI-LA/5-org.npy')
predito_dctts = np.load('samples/5.png.npy')

# unpad
predito =  predito_dctts[:y_true.shape[0]][:]
#predito = predito.reshape(-1)
#y_true = y_true.reshape(-1)
print( " Predito vs esperado R Square: ",r2_score(y_true, predito, multioutput='variance_weighted'))


y_pred = np.load('../savedir/synthesized-audios/Griff-Lim/5.npy')
y_true = predito_dctts[:y_pred.shape[0]][:]
#y_true = y_true.reshape(-1)
#y_pred = y_pred.reshape(-1)
print( " Predito vs Griff-Lim R Square: ",r2_score(y_true, y_pred, multioutput='variance_weighted'))


y_pred = np.load('../savedir/synthesized-audios/RTISI-LA/5.npy')
y_true = predito_dctts[:y_pred.shape[0]][:]
#y_true = y_true.reshape(-1)
#y_pred = y_pred.reshape(-1)
print( " Predito vs RTISI-LA R Square: ",r2_score(y_true, y_pred, multioutput='variance_weighted'))

