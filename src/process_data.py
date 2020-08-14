import numpy as np
import librosa

def extract_features_mfcc(file_name, n_mfcc=40):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccsscaled