import pandas as pd
import VP
import utils
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import models

start = time.time()
#prepare data
signals_paths = ('./teste_Guilherme/1650.1/1650.1_familiar_Electrode_Raw_Data.csv',
                './teste_Guilherme/1650.1/1650.1_novel_Electrode_Raw_Data.csv',

                './teste_Guilherme/1653.1/1653.1_familiar_Electrode_Raw_Data.csv',
                './teste_Guilherme/1653.1/1653.1_novel_Electrode_Raw_Data.csv',

                './teste_Guilherme/2091.0/2091.0_18-35-03_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2091.0/2091.0_18-45-57_ElectrodeRawData_novel2.csv',
                #'./teste_Guilherme/2091.0/2091.0_18-48-45_ElectrodeRawData_familiar3.csv',
                #'./teste_Guilherme/2091.0/2091.0_18-53-39_ElectrodeRawData_novel3.csv',

                './teste_Guilherme/2091.1/2091.1_17-22-20_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2091.1/2091.1_17-24-30_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2091.1/2091.1_17-34-52_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2091.1/2091.1_17-30-58_ElectrodeRawData_novel2.csv',
                
                #'./teste_Guilherme/2091.1/2091.1_17-40-04_ElectrodeRawData_familiar3.csv',
                #'./teste_Guilherme/2091.1/2091.1_17-38-12_ElectrodeRawData_novel3.csv',
                
                './teste_Guilherme/2091.2/2091.2_17-22-43_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2091.2/2091.2_17-20-03_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2091.2/2091.2_17-30-07_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2091.2/2091.2_17-28-09_ElectrodeRawData_novel2.csv',
                
                #'./teste_Guilherme/2091.2/2091.2_17-36-48_ElectrodeRawData_familiar3.csv',
                #'./teste_Guilherme/2091.2/2091.2_17-39-25_ElectrodeRawData_novel3.csv',
                
                './teste_Guilherme/2141.0/2141.0_20-03-29_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2141.0/2141.0_20-01-27_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2141.0/2141.0_20-09-05_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2141.0/2141.0_20-11-04_ElectrodeRawData_novel2.csv',
                
                #'./teste_Guilherme/2141.0/2141.0_20-14-32_ElectrodeRawData_familiar3.csv',
                #'./teste_Guilherme/2141.0/2141.0_20-12-43_ElectrodeRawData_novel3.csv',

                './teste_Guilherme/2141.2/2141.2_18-45-04_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2141.2/2141.2_18-46-57_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2141.2/2141.2_18-55-27_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2141.2/2141.2_18-53-38_ElectrodeRawData_novel2.csv',
                
                #'./teste_Guilherme/2141.2/2141.2_18-57-04_ElectrodeRawData_familiar3.csv',
                #'./teste_Guilherme/2141.2/2141.2_18-58-52_ElectrodeRawData_novel3.csv',

                './teste_Guilherme/2141.3/2141.3_17-54-02_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2141.3/2141.3_17-56-10_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2141.3/2141.3_18-01-46_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2141.3/2141.3_18-03-42_ElectrodeRawData_novel2.csv',
                
                #'./teste_Guilherme/2141.3/2141.3_18-07-25_ElectrodeRawData_familiar3.csv',
                #'./teste_Guilherme/2141.3/2141.3_18-05-19_ElectrodeRawData_novel3.csv',

                './teste_Guilherme/2142.4/2142.4_17-20-14_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2142.4/2142.4_17-17-17_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2142.4/2142.4_17-27-55_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2142.4/2142.4_17-26-00_ElectrodeRawData_novel2.csv',
                
                #'./teste_Guilherme/2142.4/2142.4_17-29-45_ElectrodeRawData_novel3.csv'

                './teste_Guilherme/2174.3/2174.3_12-39-03_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2174.3/2174.3_12-40-56_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2174.3/2174.3_12-46-42_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2174.3/2174.3_12-48-36_ElectrodeRawData_novel2.csv',

                './teste_Guilherme/2174.4/2174.4_12-57-26_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2174.4/2174.4_12-59-32_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2174.4/2174.4_13-04-19_ElectrodeRawData_novel2.csv',
                './teste_Guilherme/2174.4/2174.4_13-06-32_ElectrodeRawData_familiar2.csv',
                
                './teste_Guilherme/2174.6/2174.6_12-29-02_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2174.6/2174.6_12-30-56_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2174.6/2174.6_12-36-31_ElectrodeRawData_novel2.csv',
                './teste_Guilherme/2174.6/2174.6_12-38-58_ElectrodeRawData_familiar2.csv',
                
                './teste_Guilherme/2202.0/2202.0_12-59-03_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2202.0/2202.0_13-01-04_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2202.0/2202.0_13-06-40_ElectrodeRawData_novel2.csv',
                './teste_Guilherme/2202.0/2202.0_13-09-39_ElectrodeRawData_familiar2.csv',

                './teste_Guilherme/2202.1/2202.1_13-29-08_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2202.1/2202.1_13-31-04_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2202.1/2202.1_13-36-14_ElectrodeRawData_novel2.csv',
                './teste_Guilherme/2202.1/2202.1_13-38-22_ElectrodeRawData_familiar2.csv',

                './teste_Guilherme/2202.2/2202.2_13-39-01_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2202.2/2202.2_13-40-55_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2202.2/2202.2_13-46-42_ElectrodeRawData_familiar2.csv',
                './teste_Guilherme/2202.2/2202.2_13-48-35_ElectrodeRawData_novel2.csv',

                './teste_Guilherme/2202.3/2202.3_12-42-29_ElectrodeRawData_familiar1.csv',
                './teste_Guilherme/2202.3/2202.3_12-44-25_ElectrodeRawData_novel1.csv',
                './teste_Guilherme/2202.3/2202.3_12-49-41_ElectrodeRawData_novel2.csv',
                './teste_Guilherme/2202.3/2202.3_12-51-37_ElectrodeRawData_familiar2.csv'

                )

#categories: 1=familiar 0=novel
#signals_paths = utils.selectFiles()
categories, types, names = utils.categorize(signals_paths)
timestamps_paths = ('./teste_Guilherme/1650.1/1650.1_familiar_Events.csv',
                    './teste_Guilherme/1650.1/1650.1_novel_Events.csv',
                    './teste_Guilherme/1653.1/1653.1_familiar_Events.csv',
                    './teste_Guilherme/1653.1/1653.1_novel_Events.csv',
                    './teste_Guilherme/2091.0/2091.0_18-35-03_Events_familiar2.csv',
                    './teste_Guilherme/2091.0/2091.0_18-45-57_Events_novel2.csv',
                    
                    #'./teste_Guilherme/2091.0/2091.0_18-48-45_Events_familiar3.csv',
                    #'./teste_Guilherme/2091.0/2091.0_18-53-39_Events_novel3.csv',

                    './teste_Guilherme/2091.1/2091.1_17-22-20_Events_familiar1.csv',
                    './teste_Guilherme/2091.1/2091.1_17-24-30_Events_novel1.csv',
                    './teste_Guilherme/2091.1/2091.1_17-34-52_Events_familiar2.csv',
                    './teste_Guilherme/2091.1/2091.1_17-30-58_Events_novel2.csv',
                    
                    #'./teste_Guilherme/2091.1/2091.1_17-40-04_Events_familiar3.csv',
                    #'./teste_Guilherme/2091.1/2091.1_17-38-12_Events_novel3.csv',
                    
                    './teste_Guilherme/2091.2/2091.2_17-22-43_Events_familiar1.csv',
                    './teste_Guilherme/2091.2/2091.2_17-20-03_Events_novel1.csv',
                    './teste_Guilherme/2091.2/2091.2_17-30-07_Events_familiar2.csv',
                    './teste_Guilherme/2091.2/2091.2_17-28-09_Events_novel2.csv',
                    
                    #'./teste_Guilherme/2091.2/2091.2_17-36-48_Events_familiar3.csv',
                    #'./teste_Guilherme/2091.2/2091.2_17-39-25_Events_novel3.csv',
                    
                    './teste_Guilherme/2141.0/2141.0_20-03-29_Events_familiar1.csv',
                    './teste_Guilherme/2141.0/2141.0_20-01-27_Events_novel1.csv',
                    './teste_Guilherme/2141.0/2141.0_20-09-05_Events_familiar2.csv',
                    './teste_Guilherme/2141.0/2141.0_20-11-04_Events_novel2.csv',
                    
                    #'./teste_Guilherme/2141.0/2141.0_20-14-32_Events_familiar3.csv',
                    #'./teste_Guilherme/2141.0/2141.0_20-12-43_Events_novel3.csv',

                    './teste_Guilherme/2141.2/2141.2_18-45-04_Events_familiar1.csv',
                    './teste_Guilherme/2141.2/2141.2_18-46-57_Events_novel1.csv',
                    './teste_Guilherme/2141.2/2141.2_18-55-27_Events_familiar2.csv',
                    './teste_Guilherme/2141.2/2141.2_18-53-38_Events_novel2.csv',
                    
                    #'./teste_Guilherme/2141.2/2141.2_18-57-04_Events_familiar3.csv',
                    #'./teste_Guilherme/2141.2/2141.2_18-58-52_Events_novel3.csv',

                    './teste_Guilherme/2141.3/2141.3_17-54-02_Events_familiar1.csv',
                    './teste_Guilherme/2141.3/2141.3_17-56-10_Events_novel1.csv',
                    './teste_Guilherme/2141.3/2141.3_18-01-46_Events_familiar2.csv',
                    './teste_Guilherme/2141.3/2141.3_18-03-42_Events_novel2.csv',
                    
                    #'./teste_Guilherme/2141.3/2141.3_18-07-25_Events_familiar3.csv',
                    #'./teste_Guilherme/2141.3/2141.3_18-05-19_Events_novel3.csv',

                    './teste_Guilherme/2142.4/2142.4_17-20-14_Events_familiar1.csv',
                    './teste_Guilherme/2142.4/2142.4_17-17-17_Events_novel1.csv',
                    './teste_Guilherme/2142.4/2142.4_17-27-55_Events_familiar2.csv',
                    './teste_Guilherme/2142.4/2142.4_17-26-00_Events_novel2.csv',
                    
                    #'./teste_Guilherme/2142.4/2142.4_17-29-45_Events_novel3.csv'

                    './teste_Guilherme/2174.3/2174.3_12-39-03_Events_novel1.csv',
                    './teste_Guilherme/2174.3/2174.3_12-40-56_Events_familiar1.csv',
                    './teste_Guilherme/2174.3/2174.3_12-46-42_Events_familiar2.csv',
                    './teste_Guilherme/2174.3/2174.3_12-48-36_Events_novel2.csv',

                    './teste_Guilherme/2174.4/2174.4_12-57-26_Events_familiar1.csv',
                    './teste_Guilherme/2174.4/2174.4_12-59-32_Events_novel1.csv',
                    './teste_Guilherme/2174.4/2174.4_13-04-19_Events_novel2.csv',
                    './teste_Guilherme/2174.4/2174.4_13-06-32_Events_familiar2.csv',

                    './teste_Guilherme/2174.6/2174.6_12-29-02_Events_familiar1.csv',
                    './teste_Guilherme/2174.6/2174.6_12-30-56_Events_novel1.csv',
                    './teste_Guilherme/2174.6/2174.6_12-36-31_Events_novel2.csv',
                    './teste_Guilherme/2174.6/2174.6_12-38-58_Events_familiar2.csv',

                    './teste_Guilherme/2202.0/2202.0_12-59-03_Events_novel1.csv',
                    './teste_Guilherme/2202.0/2202.0_13-01-04_Events_familiar1.csv',
                    './teste_Guilherme/2202.0/2202.0_13-06-40_Events_novel2.csv',
                    './teste_Guilherme/2202.0/2202.0_13-09-39_Events_familiar2.csv',

                    './teste_Guilherme/2202.1/2202.1_13-29-08_Events_novel1.csv',
                    './teste_Guilherme/2202.1/2202.1_13-31-04_Events_familiar1.csv',
                    './teste_Guilherme/2202.1/2202.1_13-36-14_Events_novel2.csv',
                    './teste_Guilherme/2202.1/2202.1_13-38-22_Events_familiar2.csv',

                    './teste_Guilherme/2202.2/2202.2_13-39-01_Events_novel1.csv',
                    './teste_Guilherme/2202.2/2202.2_13-40-55_Events_familiar1.csv',
                    './teste_Guilherme/2202.2/2202.2_13-46-42_Events_familiar2.csv',
                    './teste_Guilherme/2202.2/2202.2_13-48-35_Events_novel2.csv',

                    './teste_Guilherme/2202.3/2202.3_12-42-29_Events_familiar1.csv',
                    './teste_Guilherme/2202.3/2202.3_12-44-25_Events_novel1.csv',
                    './teste_Guilherme/2202.3/2202.3_12-49-41_Events_novel2.csv',
                    './teste_Guilherme/2202.3/2202.3_12-51-37_Events_familiar2.csv'
                    )

accuracy_list = list()

chanel = 2


#df = utils.prepareData(signals_paths, timestamps_paths, chanel, categories, types, names, file='./Pickle/05092022_220542.pickle', meanstdev=True)


#train model
df_conjunto_1 = pd.read_csv('./CSV/07092022_121214.csv')
df_conjunto_2 = pd.read_csv('./CSV/26092022_193503.csv')
#print(df.shape)
#male_list = [1650.1, 2091.0, 2091.1, 2091.2, 2141.0, 2141.2, 2141.3, 2142.4]
#female_list = [2174.1, 2174.3, 2174.4, 2174.6, 2202.0, 2202.1, 2202.2, 2202.3, 1653.1, 2174.1, 2174.3, 2174.4, 2174.6]
#animals_2170_ = [2174.1, 2174.3, 2174.4, 2174.6]
#animals_2202_ = [2202.0, 2202.1, 2202.2, 2202.3]
#animals_214_ = [2141.0, 2141.2, 2141.3, 2142.4]
data_conjunto_1 = df_conjunto_1.drop(labels=['Label', 'Type', 'Name'], axis='columns')
data_conjunto_2 = df_conjunto_2.drop(labels=['Label', 'Type', 'Name'], axis='columns')
#data = df[df['Name'].isin([2174.4, 2174.3])]
#labels = data.loc[:, "Label"]
#data = data.drop(labels=['Label', 'Type', 'Name'], axis='columns')
data_SVM = MinMaxScaler().fit_transform(data_conjunto_1)
data_SVM = pd.DataFrame(data_SVM)
data_NN = RobustScaler().fit_transform(data_conjunto_1)
data_NN = pd.DataFrame(data_NN)
data_RF = RobustScaler().fit_transform(data_conjunto_2)
data_RF = pd.DataFrame(data_RF)
#data = RobustScaler().fit_transform(data)
#print(data)
labels_conjunto_1 = df_conjunto_1.loc[:, "Label"]
labels_conjunto_2 = df_conjunto_2.loc[:, "Label"]

models.SVM(data_SVM, labels_conjunto_1)

models.NN(data_NN, labels_conjunto_1)

#models.logistic_regression(data_NN, labels)

models.RF(data_RF, labels_conjunto_2)

#models.kNN(data_SVM, labels)





end = time.time()
print("Time to run: " + str(end - start))