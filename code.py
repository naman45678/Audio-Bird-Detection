# for pre-processing the data
# numerical processing and scientific libraries
import numpy as np
import scipy

# signal processing
from scipy.io                     import wavfile
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
import librosa

#%% Imports
import os
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from scipy.io import wavfile




#%% Initialize
Source1 = 'D:\\project\\wav'
Dest = os.system('pwd')

#%% Labels and File_add
Data = pd.read_csv('D:\\project\\ff1010bird_metadata.csv').values
Labels = Data[:,1]
Labels.reshape([-1])
File_add = [os.path.join(Source1,str(x)+'.wav') for x in Data[:,0]]

n_pos = sum(Labels)
n_neg = sum(1-Labels)

# Load the data file 
y, sr = librosa.load(File_add[1],sr=None)
y = y.reshape(1,len(y))
# Pre-emphasis filter.

# Parameters
nwin    = 256
nfft    = 1024
fs      = 16000
nceps   = 13

# Pre-emphasis factor (to take into account the -6dB/octave
# rolloff of the radiation at the lips level)
prefac  = 0.97

# MFCC parameters: taken from auditory toolbox
over    = nwin - 128

filtered_data = lfilter([1., -prefac], 1, y)
windows     = hamming(256, sym=0)
windows = windows.reshape(1,len(windows))
#
start = 0
end = nwin
frames =list()
frames.append(filtered_data[:,start:end])
while start <= 441000:
    start = end-over
    end = start + nwin
    diff = end-start
    frames.append(filtered_data[:,start:end])
windowed_frames = list()   
for i in range(len(frames)):
    windowed_frames.append(frames[i]*windows[:,:len(frames[i])])
    
magnitude_spectrum = list()
for i in range (len(frames)):
    magnitude_spectrum.append(np.abs(fft(windowed_frames[i], nfft, axis=-1)))
    
# Compute triangular filterbank for MFCC computation.

lowfreq  = 133.33
linsc    = 200/3.
logsc    = 1.0711703
fs = 44100

nlinfilt = 13
nlogfilt = 27

# Total number of filters
nfilt    = nlinfilt + nlogfilt

#------------------------
# Compute the filter bank
#------------------------
# Compute start/middle/end points of the triangular filters in spectral
# domain
freqs            = np.zeros(nfilt+2)
freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
heights          = 2./(freqs[2:] - freqs[0:-2])

# Compute filterbank coeff (in fft domain, in bins)
filterbank  = np.zeros((nfilt, nfft))

# FFT bins (in Hz)
nfreqs = np.arange(nfft) / (1. * nfft) * fs

for i in range(nfilt):
    
    low = freqs[i]
    cen = freqs[i+1]
    hi  = freqs[i+2]

    lid    = np.arange(np.floor(low * nfft / fs) + 1,
                       np.floor(cen * nfft / fs) + 1, dtype=np.int)

    rid    = np.arange(np.floor(cen * nfft / fs) + 1,
                       np.floor(hi * nfft / fs)  + 1, dtype=np.int)

    lslope = heights[i] / (cen - low)
    rslope = heights[i] / (hi - cen)

    filterbank[i][lid] = lslope * (nfreqs[lid] - low)
    filterbank[i][rid] = rslope * (hi - nfreqs[rid])
    
mscep = list()
for i in range (len(frames)):
    mscep.append(np.log10(np.dot(magnitude_spectrum[i], filterbank.T)))
    
MFCCs = dct(mscep, type=2, norm='ortho', axis=-1)[:, :nceps]
