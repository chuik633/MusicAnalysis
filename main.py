from os import listdir
import pandas as pd

# color data
from colorSampling.imageData import get_main_colors

# instrument labels
from instrumentClassification.classifier import classify_instrument


# emotino labels
from emotionClassification.classifier import classify_emotion

datadir = './movie_data/'
files = [f for f in listdir(datadir) if f.endswith('.wav')]
entries = []
for f in files:
    fpath = datadir + f
    print(f)
    # # get the colors
    # colors = get_main_colors(fpath, 100)
    # print(colors)

    # get the instrument
    instruments = classify_instrument(fpath)
    print(instruments)

    # get the emotion
    emotions = classify_emotion(fpath)
    print(emotions)
    print('-----------------------------------')

    entries.append({
        'filename': f,
        # 'colors': colors,
        'instruments': instruments,
        'emotions': emotions
    })

df = pd.DataFrame(entries)
df.to_csv(datadir + 'music_analysis.csv', index=False)

     