import joblib

from emotionClassification.myAudioData import *

emotion_features = ['amazed-suprised','happy-pleased','relaxing-calm','quiet-still','sad-lonely','angry-aggresive']
audio_feature_names = ['Mean_Acc1298_Mean_Mem40_Centroid', 'Mean_Acc1298_Mean_Mem40_Rolloff', 'Mean_Acc1298_Mean_Mem40_Flux', 'Mean_Acc1298_Mean_Mem40_MFCC_0', 'Mean_Acc1298_Mean_Mem40_MFCC_1', 'Mean_Acc1298_Mean_Mem40_MFCC_2', 'Mean_Acc1298_Mean_Mem40_MFCC_3', 'Mean_Acc1298_Mean_Mem40_MFCC_4', 'Mean_Acc1298_Mean_Mem40_MFCC_5', 'Mean_Acc1298_Mean_Mem40_MFCC_6', 'Mean_Acc1298_Mean_Mem40_MFCC_7', 'Mean_Acc1298_Mean_Mem40_MFCC_8', 'Mean_Acc1298_Mean_Mem40_MFCC_9', 'Mean_Acc1298_Mean_Mem40_MFCC_10', 'Mean_Acc1298_Mean_Mem40_MFCC_11', 'Mean_Acc1298_Mean_Mem40_MFCC_12', 'Mean_Acc1298_Std_Mem40_Centroid', 'Mean_Acc1298_Std_Mem40_Rolloff', 'Mean_Acc1298_Std_Mem40_Flux', 'Mean_Acc1298_Std_Mem40_MFCC_0', 'Mean_Acc1298_Std_Mem40_MFCC_1', 'Mean_Acc1298_Std_Mem40_MFCC_2', 'Mean_Acc1298_Std_Mem40_MFCC_3', 'Mean_Acc1298_Std_Mem40_MFCC_4', 'Mean_Acc1298_Std_Mem40_MFCC_5', 'Mean_Acc1298_Std_Mem40_MFCC_6', 'Mean_Acc1298_Std_Mem40_MFCC_7', 'Mean_Acc1298_Std_Mem40_MFCC_8', 'Mean_Acc1298_Std_Mem40_MFCC_9', 'Mean_Acc1298_Std_Mem40_MFCC_10', 'Mean_Acc1298_Std_Mem40_MFCC_11', 'Mean_Acc1298_Std_Mem40_MFCC_12', 'Std_Acc1298_Mean_Mem40_Centroid', 'Std_Acc1298_Mean_Mem40_Rolloff', 'Std_Acc1298_Mean_Mem40_Flux', 'Std_Acc1298_Mean_Mem40_MFCC_0', 'Std_Acc1298_Mean_Mem40_MFCC_1', 'Std_Acc1298_Mean_Mem40_MFCC_2', 'Std_Acc1298_Mean_Mem40_MFCC_3', 'Std_Acc1298_Mean_Mem40_MFCC_4', 'Std_Acc1298_Mean_Mem40_MFCC_5', 'Std_Acc1298_Mean_Mem40_MFCC_6', 'Std_Acc1298_Mean_Mem40_MFCC_7', 'Std_Acc1298_Mean_Mem40_MFCC_8', 'Std_Acc1298_Mean_Mem40_MFCC_9', 'Std_Acc1298_Mean_Mem40_MFCC_10', 'Std_Acc1298_Mean_Mem40_MFCC_11', 'Std_Acc1298_Mean_Mem40_MFCC_12', 'Std_Acc1298_Std_Mem40_Centroid', 'Std_Acc1298_Std_Mem40_Rolloff', 'Std_Acc1298_Std_Mem40_Flux', 'Std_Acc1298_Std_Mem40_MFCC_0', 'Std_Acc1298_Std_Mem40_MFCC_1', 'Std_Acc1298_Std_Mem40_MFCC_2', 'Std_Acc1298_Std_Mem40_MFCC_3', 'Std_Acc1298_Std_Mem40_MFCC_4', 'Std_Acc1298_Std_Mem40_MFCC_5', 'Std_Acc1298_Std_Mem40_MFCC_6', 'Std_Acc1298_Std_Mem40_MFCC_7', 'Std_Acc1298_Std_Mem40_MFCC_8', 'Std_Acc1298_Std_Mem40_MFCC_9', 'Std_Acc1298_Std_Mem40_MFCC_10', 'Std_Acc1298_Std_Mem40_MFCC_11', 'Std_Acc1298_Std_Mem40_MFCC_12']

def classify_emotion(audio_path):
    #get features
    audio_features = extract_features_from_wav(audio_path)
    #sort the features in order of the audio_feature names
    audio_features = {k: audio_features[k] for k in audio_feature_names if k in audio_features}
    audio_features = pd.DataFrame([audio_features])
    
    #load model and scaler
    model = joblib.load('./emotionClassification/data/emotion/emotion_model.pkl')
    scaler = joblib.load('./emotionClassification/data/emotion/emotion_scaler.pkl')
  

    # #scale 
    audio_features[audio_feature_names] = scaler.transform(audio_features[audio_feature_names])
    print(audio_features)

    # #predict
    predictions = model.predict(audio_features)[0]
    print(predictions)
    emotion_strings = []
    for emotion_idx in range(len(emotion_features)):
        if predictions[emotion_idx] == 1 or predictions[emotion_idx] == '1':
            emotion_strings.append(emotion_features[emotion_idx])

    return emotion_strings

# output = classify_emotion('./data/movie_music/sad.wav')
# print(output)