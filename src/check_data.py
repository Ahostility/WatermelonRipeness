import numpy as np
import librosa

np.seterr(divide='ignore', invalid='ignore')

def split(rms:list):
    threshold = np.mean(rms[0], axis=0)

    data = [0 if x < threshold else 1 for x in rms[0]]
    nzero_index = np.nonzero(data)
    diff = np.diff(nzero_index)

    bds = np.where((diff != 1))[1]

    if bds.shape[0] == 0:
        return None

    def get_index(idx):
        return nzero_index[0][idx]

    res = []
    first = 0
    for x in bds:
        res.append( (get_index(first), get_index(x)) )
        first = x + 1
    res.append( (get_index(bds[-1] + 1), get_index(-1)) )
    return res

def is_watermelon(path):
    audio, sr = librosa.load(path)
    data = librosa.feature.rms(audio, sr)

    sections = split(data)
    if sections is None:
        return False

    THRESHOLD = 0.000001

    result = False
    for s in sections:
        var = np.var(data[0][s[0]:s[1]])
        if var > THRESHOLD:
            result = True
            break

    return result
