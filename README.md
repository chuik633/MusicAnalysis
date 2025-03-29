# MusicAnalysis: Final Project Description for Machine Learning

**Background**
My thesis project is about visualizing the music of movies. For that project I’m using p5.js to illustrate things like FFTs, using colors that I extract from the scenes (part of this idea is to create an extended captions system so those that can’t hear can understand the music in movies). 

Another aspect of this project that I could see ML being really useful in is the following areas:

## Musical instrument classification: 
We did a small version of this in one of the homeworks, but it was with sound files containing one instrument. I’m interested in taking a score and breaking it down into the possible orchestra elements it’s using. 

<ins>Datasets</ins>: On kaggle there are many datasets of different instruments (for example MusicNet Dataset which has over 1 million which labels at each note time). [Dataset link](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset/data)

<ins>ML Methods</ins>: I would train a classification model on these audio files and then try to find more training data with known combined instruments (ex: 2-4 instruments at once). Ex: models like Random forests, SVMs. I would then evaluate this by training the model on 75% of the data and then testing it on the remaining 25%. I would evaluate what percentage of instruments it guessed correctly, as well as what instruments it failed to guess. I would also consider grouping types of instruments (ex: string instruments, woodwinds, etc) and then evaluating whether they guessed each of these sections for a less fine grained evaluation. 

In addition to labeling what instruments are present in the audio file, I want to get data on the percentage of sound coming from each orchestral section. 


## Music emotion classification:
One of my motivators in my thesis was how music carries a lot of plot, motifs, and emotions in movies. When you can’t hear, you miss out on these auditory cues. It could be really interesting to explore labeling emotions in sound clips. I could use this data in my visualizations. 

<ins>Datasets</ins>: I found [this article](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/1687-4722-2011-426793#Sec22) called “Multi-label classification of music by emotion.” They reference some datasets of labeled music scores which I would use. 

<ins>ML Methods</ins>: I would like to explore supervised learning from labeled datasets (classification) as well as unsupervised learning by extracting features that could be correlated to different emotion labels, clustering the audio files, and seeing what the results are. 

If I have time, I would like to do both of these and then explore how they relate. For example, do songs with more string instruments elicit sadder emotions?
