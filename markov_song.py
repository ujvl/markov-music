import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt

def adc(filehandle):
	"""function to convert analog music to digital format"""

def show_info(aname, a):
    print "Array", aname
    print "shape:", a.shape
    print "dtype:", a.dtype
    print "min, max:", a.min(), a.max()
    print

rate, data = scipy.io.wavfile.read('fur_elise.wav')
t = np.linspace(0, (len(data)-1)/rate, len(data))
show_info('data', data)
print rate
song = (data[:, 1] + data[:,0])/2

f = np.linspace(0, rate, len(data))
songfft = abs(np.fft.fft(song))
#plt.plot(f, songfft)
plt.plot(t, song)
plt.show()