import librosa
import os
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

DATASET_PATH = "train"
JSON_PATH = "train.json"
DURATION=5

def durations_histogram(dirpath, filenames):
    durations = []
    for f in filenames:
        file_path = os.path.join(dirpath, f)
        signal,sample_rate = librosa.load(file_path)
        dur = librosa.get_duration(signal,sample_rate)
        durations.append(dur)
    pd.DataFrame(durations).hist()
    
def padding_cutting(signal,sample_rate):
    #Sessizliği silme
    #signal=svt.rms_silence_filter(signal)
    signal_duration=signal.shape[0]/sample_rate
    
    if(signal_duration<DURATION):
                pad_ms = int(DURATION - signal_duration)
                pad_signal = np.zeros(sample_rate*pad_ms)
                signal=np.concatenate([signal, pad_signal])
                
    elif(signal_duration>DURATION):
                pad_ms = int(signal_duration - DURATION)
                for x in range(sample_rate*pad_ms):
                    signal = np.delete(signal,(x), axis=0)
                   
    return signal

def mfcc_mean(X):
    d = np.array([0 for x in range(13)])
    for x in range(0,13):
        d[x]=np.mean(X[x, :])
    return np.array(d)

def write_json(data,all_mfccs, file_path):
    data["MFCCs"].append(all_mfccs.T.tolist())
    data["files"].append(file_path)
                    
                    
    cat=file_path.split("_")[-1]
    cat=cat.split(".")[0]
                    
    if int(cat) > 1001508:
                    
        data["mapping"].append(cat)
        data["labels"].append(0)
    else:
                
        data["mapping"].append(cat)
        data["labels"].append(1)
def preprocess_dataset(dataset_path, json_path, num_mfcc=13, hop_length=512, n_fft=2048):
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        #MAX SUREYE KARAR VER
        #durations_histogram(dirpath, filenames)
        
        for f in filenames:
            
            all_mfccs = np.array([0 for x in range(13*DURATION)])
            file_path = os.path.join(dirpath, f)
               
            signal,sample_rate = librosa.load(file_path)
            
            signal=padding_cutting(signal, sample_rate)
            
            if(len(signal)>=sample_rate):
                signal_duration=int(signal.shape[0]/sample_rate)
                
                for x in range(signal_duration-1):
                    add_signal=signal[x*sample_rate:(x+1)*sample_rate]
                    
                    MFCCs = librosa.feature.mfcc(add_signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)
                    MFCCs=mfcc_mean(MFCCs)
                    for i in range(13):
                        #print(x," ",13*x+i)
                        all_mfccs[13*x+i]=MFCCs[i]
            
                data["MFCCs"].append(all_mfccs.T.tolist())
                data["files"].append(file_path)
                                
                                
                cat=file_path.split("_")[-1]
                cat=cat.split(".")[0]
                                
                if int(cat) > 1001508:
                                
                    data["mapping"].append(cat)
                    data["labels"].append(0)
                else:
                            
                    data["mapping"].append(cat)
                    data["labels"].append(1)
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=2)
            
def get_data_splits(data_path):
    
    X,y = load_dataset(data_path)
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test, y_train, y_test

def load_dataset(data_path):
    
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    
    return X,y    
    
preprocess_dataset(DATASET_PATH, JSON_PATH)                    
X_train, X_test, y_train, y_test = get_data_splits(JSON_PATH)

dtc_model= DecisionTreeClassifier(criterion='gini', splitter="best")
rfc_model = RandomForestClassifier(n_estimators=100, criterion='entropy')
lgr_model = LogisticRegression(n_jobs=-1, random_state=0)
nb_model = BernoulliNB()
knn_model = KNeighborsClassifier(n_jobs=-1, n_neighbors=3, p=1)
svm_model = SVC(kernel='linear')

models= []

models.append(("DTC", dtc_model))
models.append(("RFC", rfc_model))
models.append(("LGR", lgr_model))
models.append(("NB", nb_model))
models.append(("KNN", knn_model))
models.append(("SVM", svm_model))



for i, model in models:
   
    fmodel = model.fit(X_train,y_train)
    
    preds = fmodel.predict(X_test)
    scores = cross_val_score(fmodel, X_train, y_train)
    confusion_matrix = metrics.confusion_matrix(y_test, preds)
    classification = classification_report(y_test, preds)


    print('\n============================== {} Algoritması ==============================\n'.format(i))
    print ("\nCross Validation Mean Score:" , scores.mean())
    print("\nConfusion matrix: ")
    print(confusion_matrix)
    print("Classification report:" "\n", classification) 
