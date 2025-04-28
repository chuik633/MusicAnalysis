import librosa
import numpy as np
from os import listdir
import pandas as pd
#windowsize
mem_size = 40

# put it all together in a function
def compute_feature_aggregations(feature_list,feature_name):
    feature_array = np.array(feature_list)

    chunks = [
        feature_array[i:i+mem_size]
        for i in range(0, len(feature_array) - mem_size + 1, mem_size)
    ]

    chunk_means = [np.mean(chunk) for chunk in chunks]
    chunk_stds = [np.std(chunk) for chunk in chunks]

    results = {
        "Mean_Acc1298_Mean_Mem40_"+feature_name:float(np.mean(chunk_means)) ,
        "Mean_Acc1298_Std_Mem40_"+feature_name: float( np.mean(chunk_stds)),
        "Std_Acc1298_Mean_Mem40_"+feature_name: float( np.std(chunk_means)),
        "Std_Acc1298_Std_Mem40_"+feature_name:  float( np.std(chunk_stds)),
    }

    return results

def extract_features_from_wav(filename):
    y, sr = librosa.load(filename)
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    S = np.abs(librosa.stft(y))
    S_norm = librosa.util.normalize(S, axis=0)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1)**2, axis=0))

  
    data = {
        # 'filename':filename,
        **compute_feature_aggregations(centroids, "Centroid"),
        **compute_feature_aggregations(rolloffs, "Rolloff"),
        **compute_feature_aggregations(flux, "Flux"),
    }

    for i in range(13):
        name = "MFCC_"+str(i)
        data = {**compute_feature_aggregations(mfccs[i],name), **data}

    return data

def audio_directory_to_df(audio_dir):
    files = [audio_dir+f for f in listdir(audio_dir) if f.endswith('.wav')]
    data = []
    for f in files:
        entry = extract_features_from_wav(f)
        data.append(entry)
    df = pd.DataFrame(data)
    return df