from os import listdir
import pandas as pd

# color data
from colorSampling.imageData import get_main_colors

# instrument labels
from instrumentClassification.classifier import classify_instrument

# emotino labels
from emotionClassification.classifier import classify_emotion


datadir = './data/movies/'
#get all the folders names in this directory
film_folders = [f for f in listdir(datadir) if not f.startswith('.') and "." not in f]

def processMovie(movie_folder):
    # get all the files in the folder
    audio_paths = [datadir + movie_folder +'/audios/'+f for f in listdir(datadir + movie_folder +'/audios/') if f.endswith('.wav')]
    image_paths = [datadir + movie_folder +'/images/'+f for f in listdir(datadir + movie_folder +'/images/') if f.endswith('.png')]
    entries = []
    for audio_file, image_file in zip(audio_paths, image_paths):
        # get the colors
        colors = get_main_colors(image_file, 100)
        # print(colors)

        # get the instrument
        instruments = classify_instrument(audio_file)
        # print(instruments)

        # get the emotion
        emotions = classify_emotion(audio_file)
        # print(emotions)
        print('-----------------------------------')

        entries.append({
            'movie': movie_folder,
            'colors': colors,
            'instruments': instruments,
            'emotions': emotions
        })

    return entries

data = []
for film_folder in film_folders:
    print(film_folder)
    new_entries = processMovie(film_folder)
    data.extend(new_entries)
    print('-----------------------------------')


# save the data
from sklearn.preprocessing import MultiLabelBinarizer
df = pd.DataFrame(data)
df.to_csv('./data/music_analysis.csv', index=False)
mlb_instruments = MultiLabelBinarizer()
instruments_encoded = pd.DataFrame(mlb_instruments.fit_transform(df['instruments']),
                                   columns=mlb_instruments.classes_)

mlb_emotions = MultiLabelBinarizer()
emotions_encoded = pd.DataFrame(mlb_emotions.fit_transform(df['emotions']),
                                columns=mlb_emotions.classes_)


df_encoded = pd.concat([df[['movie']], instruments_encoded, emotions_encoded], axis=1)
df_encoded.to_csv('./data/music_analysis_encoded.csv', index=False)