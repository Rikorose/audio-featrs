import sys
import librosa
import numpy as np


if __name__ == '__main__':
    y, sr = librosa.load(sys.argv[1])
    print(sr)
    D = np.abs(librosa.stft(y))
    print(D.min(), D.max())
    print(D.shape)
    #D = np.square(D)
    #print(D.min(), D.max())
    D = librosa.amplitude_to_db(D)
    print(D.min(), D.max())
