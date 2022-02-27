import os
import pandas as pd
import numpy as np
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class RAVDESS_Dataset:
    def __init__(self):
        # Paths for data.
        self.path = "/home/andrealombardi/speech_emotion_recognition/Ravdess/audio_speech_actors_01-24/"

    def create_file_path_list(self):

        ravdess_directory_list = os.listdir(self.path)

        file_emotion = []
        file_path = []
        for dir in ravdess_directory_list:
            # as their are 20 different actors in our previous directory we need to extract files for each actor.
            actor = os.listdir(self.path + dir)
            for file in actor:
                part = file.split('.')[0]
                part = part.split('-')
                # third part in each file represents the emotion associated to that file.
                file_emotion.append(int(part[2]))
                file_path.append(self.path + dir + '/' + file)
                
        # dataframe for emotion of files
        emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

        # dataframe for path of files.
        path_df = pd.DataFrame(file_path, columns=['Path'])
        Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

        # changing integers to actual emotions.
        Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
        Ravdess_df.head()

        # creating Dataframe using all the 4 dataframes we created so far.
        #data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
        data_path = pd.concat([Ravdess_df], axis = 0)
        data_path.to_csv("data_path.csv",index=False)
        data_path.head()

        #plt.title('Count of Emotions', size=16)
        #sns.countplot(data_path.Emotions)
        #plt.ylabel('Count', size=12)
        #plt.xlabel('Emotions', size=12)
        #sns.despine(top=True, right=True, left=False, bottom=False)
        #plt.show()

    def extract_features(self, data):
        # taking a random example and checking for its sample_rate.
        _ , sample_rate = librosa.load("/home/andrealombardi/speech_emotion_recognition/Ravdess/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav")

        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally

        # Spectral constrat
        spect_contr = np.mean(librosa.feature.spectral_contrast(data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, spect_contr))

        return result

    def noise(self, data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def get_features(self, path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        
        # without augmentation
        res1 = self.extract_features(data=data)
        result = np.array(res1)
        
        # data with noise
        noise_data = self.noise(data)
        res2 = self.extract_features(noise_data)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        new_data = self.stretch(data)
        data_stretch_pitch = self.pitch(new_data, sample_rate)
        res3 = self.extract_features(data_stretch_pitch)
        result = np.vstack((result, res3)) # stacking vertically
        
        return result

    def create_dataset(self):

        if os.path.exists("ravdess_dataset_augmented.csv")==False:
            print("Creating dataset...\n")

            if os.path.exists("data_path.csv")==False:
                self.create_file_path_list()
        
            data_path = pd.read_csv('data_path.csv')

            X, Y = [], []
            i = 0

            for path, emotion in zip(data_path.Path, data_path.Emotions):
                pth = path
                feature = self.get_features(pth)
                if i%100 == 0:
                    print(str(i) + " processed elements.")

                for element in feature:
                    X.append(element)
                    # appending emotion 3 times as we have made 2 augmentation techniques on each audio file.
                    Y.append(emotion)
                i+=1

            dataset = pd.DataFrame(X)
            dataset['labels'] = Y
            dataset.to_csv('ravdess_dataset_augmented.csv', index=False)
            dataset.head()

            return X, Y

        
        else: #if the dataset has already been created we just read the csv file and return it
            print("Getting dataset...\n")

            dataset = pd.read_csv('ravdess_dataset_augmented.csv')
            X = dataset.iloc[: ,:-1].values
            Y = dataset['labels'].values

            return X, Y
