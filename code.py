import numpy as np
import scipy

from scipy.io                     import wavfile
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
