import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import librosa
import numpy as np

def get_wav(language_num):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''
    y, sr = librosa.load('../audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=24000, scale=True))

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=24000, n_mfcc=13))

def filter_df(df):
    '''
    Function to filter audio files based on df columns
    df column options: [age,age_of_english_onset,age_sex,birth_place,english_learning_method,
    english_residence,length_of_english_residence,native_language,other_languages,sex]
    :param df (DataFrame): Full unfiltered DataFrame
    :return (DataFrame): Filtered DataFrame
    '''

    # Example to filter arabic, mandarin, and english and limit to 73 audio files
    arabic,arabicy = [],[]
    mandarin,mandariny = [],[]
    english,englishy = [],[]
    
    for i in range(79):
        english.append(to_mfcc(get_wav("english"+str(i+1))))
        englishy.append("english")
        mandarin.append(to_mfcc(get_wav("mandarin"+str(i+1))))
        mandariny.append("mandarin")
        arabic.append(to_mfcc(get_wav("arabic"+str(i+1))))
        arabicy.append("arabic")

    val = english + arabic + mandarin
    val2 = englishy + arabicy + mandariny
    df = {'wav':val,'native_language':val2}
    return df

def split_people(df,test_size=0.2):
    '''
    Create train test split of DataFrame
    :param df (DataFrame): Pandas DataFrame of audio files to be split
    :param test_size (float): Percentage of total files to be split into test
    :return X_train, X_test, y_train, y_test (tuple): Xs are list of df['language_num'] and Ys are df['native_language']
    '''
    return train_test_split(df['wav'],df['native_language'],test_size=test_size,random_state=1234)


if __name__ == '__main__':
    '''
    Console command example:
    python bio_data.csv
    '''
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    # print(df.values[0])
    filtered_df = filter_df(df)
    print(split_people(filtered_df))
