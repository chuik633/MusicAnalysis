
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import json
import shutil


from data_creation.sceneMetaData import emotionSceneLinks

"""
inputting a movie file, it then:
1. if its a youtube link, it downloads it to a video
2. splits the video in n_samples videos, and corresponding audio files and images
3. gets the color information for each scene imageSceneData.json
4. gets the audio data for each scene and saves it in audioSceneData.json
5. gets the caption data
6. saves a compiled scene data
"""
def clearDirectories(mainDir):
    for dir_name in ['videos', 'images', 'audios']:
        dir_path = os.path.join(mainDir, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            # print('deleted dir', dir_name)

def getData(name, numSamples = 20, youtubeLink = False, captions = False):
    dataDir = './data/movies/'+name+'/'
    # print(dataDir)

    if os.path.exists(dataDir):
        return True
    # clearDirectories(dataDir)
    os.makedirs(dataDir, exist_ok=True)
    
    #1. if its a youtube link, it downloads it to a video
    if youtubeLink != False:
        try:
            command = [
               'yt-dlp',
                '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                '--merge-output-format', 'mp4',
                '-o', dataDir + "video.mp4",
                youtubeLink
            ]
            # command = ['yt-dlp', '-f', 'bestvideo+bestaudio', '-o', dataDir + "video.mp4", youtubeLink]
            result = subprocess.run(command, capture_output=True, text=True, check=True)

        except subprocess.CalledProcessError as e:
            # print("Error downloading video:", name)
            shutil.rmtree(dataDir)
            return False

    #2. splits the video in n_samples videos, and corresponding audio files and images
    
    command = ['node', './data_creation/processVideo.js',dataDir, str(numSamples)]
    print(command)
    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)
        print("Split video successfully")
        print(result)
    except subprocess.CalledProcessError as e:
        # print("Error downloading video:", name)
        shutil.rmtree(dataDir)
        return False
    return True
    
    

for emotion in emotionSceneLinks.keys():
    failed = []
    print("processing emotion",emotion)
    for movieName in emotionSceneLinks[emotion].keys():
        url = emotionSceneLinks[emotion][movieName]["url"]
        output = getData(movieName, 1, youtubeLink=url, captions = False)
        if output == False:
            #remove that entry from emotionSceneLinks
            failed.append(movieName)
    for movieName in failed:
        print("removing entry",movieName)
        emotionSceneLinks[emotion].pop(movieName)