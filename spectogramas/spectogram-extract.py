
from matplotlib import pyplot as plt
import numpy as np
import librosa
from librosa import  display
import os

vocoders = ['RTISI-LA','Griff-Lim']
for voco in vocoders:
    diretory = os.path.join('../synthesized-audios/',voco)
    arquivos = ['5.wav','5-org.wav','3.wav','3-org.wav']
    for i in arquivos:
        y, sr = librosa.load(os.path.join(diretory,i))
        D = np.abs(librosa.stft(y))

        display.specshow(librosa.amplitude_to_db(D,ref=np.max), y_axis='log', x_axis='time')
        plt.title('Espectrograma STFT')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(os.path.join(voco,'spectrograma-'+voco+'-'+i.replace('.wav','')+'.png'))
        plt.cla()   # Clear axis
        plt.clf()
        #plt.show()