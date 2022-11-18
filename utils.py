from asyncio.windows_events import NULL
import pandas as pd
import VP
import featureExtraction as fe
import preProcess as pp
import featureReduction as fr
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import tkinter,tkinter.filedialog
from tkinter import messagebox
from datetime import datetime

def categorize(strings):
    #categories: 1=familiar 0=novel
    categories = []
    types = []
    names = []
    with open('repository/ratos.txt') as f:
        lines = f.readlines()
    
    splited_lines = []
    for line in lines:
        splited_line = line.split(",")
        splited_lines.append(splited_line)

    for string in strings:
        if(string.find('familiar')!=-1):
            categories.append(1)
        elif(string.find('novel')!=-1):
            categories.append(0)
        else:
            print("Error in categorization!")
        
        for line in splited_lines:
            if(string.find(line[0])!=-1):
                types.append(line[2])
                names.append(line[0])

    categories = tuple(categories)
    types = tuple(types)
    names = tuple(names)
    return categories, types, names


def convert_voltage(df, chanel):

    values = df.iloc[:, chanel]

    for i, value in enumerate(values):
        values[i] = value*10**(-6)
    
    df.iloc[:,chanel] = values

    return df


def readFiles(files, isTimestamps=False):

    df_list = list()

    if type(files) != tuple:
        files = (files,)

    for file in files:
        
        if(isTimestamps):
            df = pd.read_csv(file, header=5, usecols=[0])
        else:
            df = pd.read_csv(file, header=5)
        
        df_list.append(df)

    return df_list

def extract_epochs(signals, timestamps, chanel, categories, types, names):

    epochs_list = list()
    i = 1
    for signal, timestamp, category, tipo, name in zip(signals,timestamps, categories, types, names):
        print("Signal " + str(i) + "/" + str(len(signals)))
        start = time.time()
        signal = signal.dropna()
        epoch_df = VP.get_epochs(signal,timestamp, chanel)
        print(name)
        if epoch_df is not None:
            print("Len"+str(len(epoch_df)))
            epoch_df.insert(len(epoch_df.columns), 'Label', category)
            epoch_df.insert(len(epoch_df.columns), 'Type', tipo)
            epoch_df.insert(len(epoch_df.columns), 'Name', name)
            epochs_list.append(epoch_df)
        i = i+1
        
        end = time.time()
        print(end - start)
    return epochs_list

def selectFeatures(features, num=40):

    size = np.arange(0,len(features), 1)
    x = random.choices(size, k=num)

    random_selected_features = features.iloc[x]
    random_selected_features.index = range(len(random_selected_features))

    return random_selected_features




def prepareData(signals_paths, timestamps_paths, chanel, categories, types, names, file=None, save=False, meanstdev=False):

    #date and time for file saving purposes
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    print(dt_string)
    print('Preparing Data...')

    if(file == None):
        print('Step 1/9: Reading Signals...')
        signals = readFiles(signals_paths)
        print('Step 1/9 Done')
        frequency_checked_signals = list()
        for signal in signals:
            frequency_checked_signal = pp.checkFrequency10k(signal, chanel)
            frequency_checked_signals.append(frequency_checked_signal)

        print('Step 2/9: Reading Timestamps...')
        timestamps = readFiles(timestamps_paths, isTimestamps=True)
        print('Step 2/9 Done')
        
        print('Step 3/9: Parcing Epochs...')
        epochs_list = extract_epochs(signals, timestamps, chanel, categories, types, names)
        print('Step 3/9 Done')
        if save == True:
            file_path = "./Pickle/" + dt_string + ".pickle"
            saveObject(epochs_list, file_path)
    else:
        epochs_list = loadObject(file)

    #print('Step 4/9: PreProcessing...')
    #preProcessed_epochs_list = list()
    #for epochs in epochs_list:

    #    data = epochs.drop(labels=['Label', 'Type', 'Name'], axis='columns')
    #    info = epochs[['Label', 'Type', 'Name']]
    #    preProcessed_epochs = pp.preProcess(data)
        #for i in range(len(preProcessed_epochs)):
            #print(preProcessed_epochs.iloc[i,:])
            #plotp100(preProcessed_epochs.iloc[i,:])
    #    preProcessed_epochs = pd.concat([preProcessed_epochs, info], axis=1)
    #    preProcessed_epochs_list.append(preProcessed_epochs)
    #print('Step 4/9 Done')
    preProcessed_epochs_list = epochs_list
    print('Step 5/9: Extracting Features...')
    features_list = list()
    for preProcessed in preProcessed_epochs_list:

        data = preProcessed.drop(labels=['Label', 'Type', 'Name'], axis='columns')
        info = preProcessed[['Label', 'Type', 'Name']]
        if(meanstdev):
            features = fe.extractWelch(data)
            features = fe.welch_mean_std(features)
        else:
            features = fe.extractWelch(data)
        #features = fe.extract_p100(preProcessed)
        features = pd.concat([features, info], axis=1)
        features_list.append(features)
    print('Step 5/9 Done')

    #selected_features_list = list()
    #for features in features_list:

        #selected_features = selectFeatures(features)
        #print(selected_features)
    #    data = features.drop(labels=['Type', 'Name'], axis='columns')
    #    cor = data.corr()
    #    cor_target = abs(cor["Label"])
    #    relevant_features = cor_target[cor_target>0.5]
    #    selected_features_list.append(relevant_features)

    print('Step 6/9: Feature Reduction...')
    reduced_list = list()
    for features in features_list:
        data = features.drop(labels=['Label', 'Type', 'Name'], axis='columns')
        info = features[['Label', 'Type', 'Name']]
        print(len(data.columns))
        if len(data.columns)>=15:
            reduced = fr.SVD(data)
            reduced = pd.concat([reduced, info], axis=1)
            reduced_list.append(reduced)
        else:
            reduced_list = features_list
            print("Not enough features for reduction. Nr needs to be greater than 15.")
            break
    print('Step 6/9 Done')
    #print(reduced_list)
    #reduced_list = features_list
    #print('Step 7/9: Categorizing...')
    #categorized = list()
    #for data, category, tipo, name in zip(reduced_list, categories, types, names):

    #    data.insert(len(data.columns), 'Label', category)
    #    data.insert(len(data.columns), 'Type', tipo)
    #    data.insert(len(data.columns), 'Name', name)
    #    categorized.append(data)
    #print('Step 7/9 Done')

    print('Step 8/9: Concatenating Data...')
    preparedData = pd.concat(reduced_list, ignore_index=True)
    print('Step 8/9 Done')

    print('Step 9/9: Saving Dataset...')
    #preparedData.to_csv('./teste' + str(chanel) + '.csv')
    file_path = "./CSV/" + dt_string + ".csv"
    preparedData.to_csv(file_path)
    print('Done Preprocessing')

    return preparedData

def plotp100(epoch):
    
    p100stuff = fe.p100amplitude(epoch)
    x =np.arange(0, 75, 75/len(epoch))
    plt.plot(x, epoch)
    
    if p100stuff != None:
        first_peak = p100stuff[1]
        print(p100stuff[0]*10**(-6))
        first_minimum = p100stuff[2]
        
        plt.plot(x[first_peak], epoch[first_peak], 'ro')
        plt.plot(x[first_minimum], epoch[first_minimum], 'yo')

    
    plt.show()

def saveObject(object, file):

    file_to_store = open(file, "wb")
    pickle.dump(object, file_to_store)
    file_to_store.close()

def loadObject(file):

    file_to_read = open(file, "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()

    return loaded_object
