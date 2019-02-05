
from matplotlib import pyplot as plt
import numpy as np
import librosa
from librosa import  display
import os
from utils import load_spectrograms
from hyperparams import Hyperparams as hp


# import the necessary packages
from skimage.measure import compare_ssim as ssim

from sklearn.metrics import r2_score
import numpy
from  math import log2,sqrt,log,log10
import math

def RMSEf0(fr,fs):
	return 1200*sqrt((log2(fr.mean())-log2(fs.mean()))**2)
def RMSE_Spec(X,Y):
	'''Reference: Tamamori, A., Hayashi, T., Kobayashi, K., Takeda, K. 
	   and Toda, T., 2017, August. Speaker-dependent WaveNet 
	   vocoder. In Proc. Interspeech (Vol. 2017, pp. 1118-1122).'''
	   
	F=len(X)
	rmse = 0
	for f in range(F):
		print(Y[f])
		rmse+=(20*np.log10(np.abs(Y[f])/np.abs(X[f])))**2
		#print(rmse)
	rmse = rmse/F
	rmse = np.sqrt(rmse)
	return  rmse


def RMSE(predictions, targets):
	rmse=np.sqrt(((predictions - targets) ** 2).mean())
	return rmse



def MCD(fr,fs):
	'''Reference: Tamamori, A., Hayashi, T., Kobayashi, K., Takeda, K. 
	   and Toda, T., 2017, August. Speaker-dependent WaveNet 
	   vocoder. In Proc. Interspeech (Vol. 2017, pp. 1118-1122).'''
	differences = fr - fs 
	differences_squared = differences ** 2 
	mean_of_differences_squared = differences_squared.mean()
	MCD = (10/log(10))*sqrt(2*(mean_of_differences_squared))
	return MCD

def psnr(img1, img2):
    #Peak Signal-to-Noise Ratio (PSNR)
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 'INF'
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def FrameRMSE(X_data,Y_data):
	X = []
	Y=[]
	if X_data.shape[0] < Y_data.shape[0]:
		Y_data=Y_data[:X_data.shape[0]]
	else:
		X_data=X_data[:Y_data.shape[0]]
	print(X_data.shape,Y_data.shape)
	#get frames
	for i in range(X_data.shape[1]):
		X.append(X_data[:,i])
		Y.append(Y_data[:,i])

	lenx= len(X)
	mse =0
	for i in range(lenx):
		mse = mse+RMSE(X[i],Y[i])
	return mse/lenx



esperado_dir = '../avaliacao-subjetiva/Esperado/'
exp_dir = '../avaliacao-subjetiva/Preditos/'

exp_list = list(os.listdir(exp_dir))
esperado_list = list(os.listdir(esperado_dir))
esp_mag = []
esp_db= []
for i in esperado_list:
    if i[-4:] == '.wav':
        file_id = i[:-4]
        _,_,mag= load_spectrograms(os.path.join(esperado_dir,i))
        db = librosa.amplitude_to_db(mag,ref=np.max)
        display.specshow(db, y_axis='log', x_axis='time')
        save_img_dir=os.path.join(esperado_dir,i.replace('.wav','.png'))
        esp_mag.append([file_id,mag])
        esp_db.append([file_id,db])
        
        plt.title('Espectrograma STFT')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        
        plt.savefig(save_img_dir)


        plt.cla()   # Clear axis
        plt.clf()


results_list=[]
for pasta in exp_list:
    arq_pasta= list(os.listdir(os.path.join(exp_dir,pasta)))
    for i in arq_pasta:
        if i[-4:] == '.wav':
            file_id = i[:-4]
            _,_,mag= load_spectrograms(os.path.join(exp_dir,pasta,i))
            db = librosa.amplitude_to_db(mag,ref=np.max)
            display.specshow(db, y_axis='log', x_axis='time')
            plt.title('Espectrograma STFT')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir,pasta,i.replace('.wav','.vocoder.png')))
            plt.cla()   # Clear axis
            plt.clf()
            for j in range(len(esp_mag)):
                if esp_mag[j][0]== file_id:
                    print(esp_mag[j][0],file_id,pasta)
                    dimshape= esp_db[j][1].shape[0] 
                    dbshape = db.shape[0]
                    if dbshape >dimshape:
                        db = db[:dimshape][:]
                    else:
                        esp_db[j][1] = esp_db[j][1][:dbshape][:]
                    dimshape= esp_mag[j][1].shape[0] 
                    magshape = mag.shape[0]
                    if magshape >dimshape:
                        mag = mag[:dimshape][:]
                    else:
                        esp_mag[j][1] = esp_mag[j][1][:magshape][:]
                    
                    
                    #print("Esperado  vs. ",pasta,': R2 db: ',r2_score(esp_db[j][1], db, multioutput='variance_weighted'),'R2 mag: ',r2_score(esp_mag[j][1], mag, multioutput='variance_weighted'),' SSIM: ',ssim(esp_db[j][1], db),'RMSE db: ',FrameRMSE(esp_db[j][1], db),'RMSE mag: ',FrameRMSE(esp_mag[j][1], mag))
                    results_list.append([pasta,r2_score(esp_db[j][1], db, multioutput='variance_weighted'),r2_score(esp_mag[j][1], mag, multioutput='variance_weighted'),ssim(esp_db[j][1], db),FrameRMSE(esp_db[j][1], db),FrameRMSE(esp_mag[j][1], mag)])
values =[]
for val in results_list:
    if val[0] not in values:
        values.append(val[0])


for exp in values:
    median_results = [0,0,0,0,0]
    for val in results_list:
        if val[0] ==exp:
            for i in range(5):
                median_results[i]+=val[i+1]
    lenmedian= len(median_results)
    print("Esperado  vs. ",exp,': R2 db: ',median_results[0]/lenmedian,'R2 mag: ',median_results[1]/lenmedian,' SSIM: ',median_results[2]/lenmedian,'RMSE db: ',median_results[3]/lenmedian,'RMSE mag: ',median_results[4]/lenmedian)

            



            