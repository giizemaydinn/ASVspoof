import os
import shutil
import librosa
import json
import numpy as np
import speaker_verification_toolkit.tools as svt

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

import tensorflow.keras as keras
from keras.layers import  Dropout
from keras. layers import Conv2D, MaxPooling2D
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform

DURATION = 5 

def move_file(dataset_path, year, protocol):
    filename=dataset_path+".txt"
    file = open (filename, "r")
    filelist = file.readlines()
    file.close()
   
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            
            for f in filenames:
                f=f.split(".wav")[0] #klasordeki
                
                for x in filelist:
                    
                    file=x.split(" ")[0] #txt'deki
                    
                    if(f == file):
                        spoof=x.split(" ")[-1]
                        
                        if(spoof.startswith("s")):
                            current_path = dirpath+ "/"+f+".wav"
                            move_path = year+"/"+protocol+"/spoof/" + f+".wav"
                            shutil.move(current_path, move_path)
                        else:
                            current_path = dirpath+ "/"+f+".wav"
                            move_path = year+"/"+protocol+"/genuine/" + f+".wav"
                            shutil.move(current_path, move_path)  

def durations_histogram(dirpath, filenames):
    durations = []
    for f in filenames:
        file_path = os.path.join(dirpath, f)
        signal,sample_rate = librosa.load(file_path)
        dur = librosa.get_duration(signal,sample_rate)
        durations.append(dur)
    pd.DataFrame(durations).hist()
    
def padding_cutting(signal,sample_rate):
    
    signal_duration=signal.shape[0]/sample_rate
    
    if(signal_duration<DURATION):
                pad_ms = int(DURATION - signal_duration)
                pad_signal = np.zeros(sample_rate*pad_ms)
                signal=np.concatenate([signal, pad_signal])
                
    elif(signal_duration>DURATION):
                pad_ms = int(signal_duration - DURATION)
                for x in range(sample_rate*pad_ms):
                    try:
                        signal = np.delete(signal,(x), axis=0)
                    except:
                        a=0
                  
                        
                   
    return signal
      
def mfcc_mean(X):
    d = [0 for x in range(13)]
    for x in range(0,13):
        d[x]=np.mean(X[x, :])
        
    return np.array(d)
                          
def preprocess_dataset(dataset_path, json_path, num, num_mfcc=13, hop_length=512, n_fft=2048):
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }
    
    for t in range(2):
        if t == 0:
            path=dataset_path+"\\genuine"
        else:
            path=dataset_path+"\\spoof"
            
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
                
                #MAX SUREYE KARAR VER
                #durations_histogram(dirpath, filenames)
                
                for a, f in enumerate(filenames):
                    
                    if(a==num and num != 0):
                        print(a)
                        break
                    all_mfccs = [[0 for a in range(13)] for y in range(DURATION)]
                  
                    
                    file_path = os.path.join(dirpath, f)
                       
                    signal,sample_rate = librosa.load(file_path)
                    
                    signal=padding_cutting(signal, sample_rate)
                    
                    if(len(signal)>=sample_rate):
                        signal_duration=int(signal.shape[0]/sample_rate)
                        
                        for x in range(signal_duration-1):
                            add_signal=signal[x*sample_rate:(x+1)*sample_rate]
                            
                            
                            MFCCs = librosa.feature.mfcc(add_signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)
                           
                            MFCCs=mfcc_mean(MFCCs)
                          
                            for a in range(13):
                                all_mfccs[x][a]=MFCCs[a]
                                    
                        
                        
                        data["MFCCs"].append(np.array(all_mfccs).tolist())
                        data["files"].append(file_path)
                                        
                                        
                        cat=file_path.split("_")[-1]
                        cat=cat.split(".")[0]
                             
                        data["mapping"].append(cat)
                        data["labels"].append(t)
                       
                with open(json_path, "w") as fp:
                    json.dump(data, fp, indent=2) 
        
def load_dataset(data_path):
    
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
   
    #X= X.reshape(X.shape[0],5,13)
    
    return X,y  

def identity_block(X, f, filters):

    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)


    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)


    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

def convolutional_block(X, f, filters, s = 2):


    F1, F2, F3 = filters
    

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s))(X) # 1,1 is filter size
    X = BatchNormalization(axis = 3)(X)  # normalization on channels
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)


    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)


    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)


    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

def ResNet50(input_shape, classes):

    X_input = Input(input_shape)


    X = ZeroPadding2D((3, 3))(X_input) 

    X = Conv2D(64, (7, 7), strides=(2, 2))(X) 
    X = BatchNormalization(axis=3)(X) 
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
   
   
    X = identity_block(X, 3, [64, 64, 256]) 
  
  
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])


    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    

    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
                    
#move_file("train-2017", "2017", "train")         #dosyaları spoof/bonafide ayırma

#preprocess_dataset("2017\\train", "train-2017.json", 1500)  #dosyaları mfcc ile jsona yazma

#train üzerinde deneme  
"""X, y = load_dataset("train-2019.json")                      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)"""


#test üzerinde deneme 
X_train, y_train = load_dataset("train-2017.json")  
X_test, y_test = load_dataset("test-2017.json")


#vektörel halde kullanmak için
x_train= np.reshape(X_train,(X_train.shape[0], X_train.shape[1]*X_train.shape[2])) 
x_test= np.reshape(X_test,(X_test.shape[0], X_test.shape[1]*X_test.shape[2])) 


"""dtc_model= DecisionTreeClassifier(criterion='gini', splitter="best")
rfc_model = RandomForestClassifier(n_estimators=100, criterion='entropy')
lgr_model = LogisticRegression(n_jobs=-1, random_state=0)
nb_model = BernoulliNB()
knn_model = KNeighborsClassifier(n_jobs=-1, n_neighbors=3, p=1)"""
svm_model = SVC(kernel='linear')


models= []

"""models.append(("DTC", dtc_model))
models.append(("RFC", rfc_model))
models.append(("LGR", lgr_model))
models.append(("NB", nb_model))
models.append(("KNN", knn_model))"""
models.append(("SVM", svm_model))



for i, model in models:
   
    fmodel = model.fit(x_train,y_train)
    
    preds = fmodel.predict(x_test)
    scores = cross_val_score(fmodel, x_train, y_train)
    confusion_matrix = metrics.confusion_matrix(y_test, preds)
    classification = classification_report(y_test, preds)


    print('\n============================== {} Algoritması ==============================\n'.format(i))
    print ("\nCross Validation Mean Score:" , scores.mean())
    print("\nConfusion matrix: ")
    print(confusion_matrix)
    print("Classification report:" "\n", classification)
    
model = keras.Sequential([

        keras.layers.Flatten(input_shape=x_train[0].shape),
      
        keras.layers.Dense(65, activation='relu'),

        keras.layers.Dense(130, activation='relu'),

    
        keras.layers.Dense(65, activation='relu'),

      
        keras.layers.Dense(2, activation='sigmoid')
    ])

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100)


X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]


model = Sequential() 

model.add(BatchNormalization(input_shape=(X_train[0].shape)))

model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=X_train[0].shape))

 
model.add(Conv2D(128, (3, 3), activation='relu'))


model.add(Dropout(0.25))


model.add(Flatten())


model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))


model.add(Dense(2, activation='softmax'))

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=100)

model = ResNet50(input_shape = X_train[0].shape, classes = 2)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=100)