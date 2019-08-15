# most of the code is copied from GettingData-LiveGraph.py
# this program make real time prediction based on trained model.
# check model-saving-test.ipybd for prediction

# overflow
# load trained model.h5
# while i <= sets
# get data
# make predction
# contain data into pd dataframe
# save data


# params
sleepTime = 0.001 # for plt.pause [s]
Yscope = 0.10 # min and max value of y axis
skip = 10 # the number of data which will be skipped until next plot
modelname = '2019-08-09-17-04-36-none-R55' # select model based on training data
modelname = "model-{}".format(modelname)
modelname = 'models/{}.h5'.format(modelname)

# Library
import ConsoleApplication1 as con
import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import datetime
import os
import glob
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time
import serial
import datetime
import sklearn
from sklearn import svm, preprocessing
import tensorflow as tf
from tensorflow import keras
print("Library imported")
print("tf.version", tf.__version__) # should be 1.14.0



# creating new dataframe (check Getting-data.py to see how to load previous data)
rows=0
data=pd.DataFrame()
dataraw=pd.DataFrame()
dataref=pd.DataFrame()
labellist=[]
labelnum=np.array([])



# preparing for prediction
# loading trained model
print("Loading model ...")
new_model = keras.models.load_model(modelname)
print(new_model.summary())
print("\n\n")



# measuring
while True:
    # LABEL NAME
    print("label name: (finished -> exit)")
    labelname=input()
    # finished
    if labelname=="exit":
        data=data.append(pd.Series(labelnum),ignore_index=True)
        dataraw=dataraw.append(pd.Series(labelnum),ignore_index=True)
        dataref=dataref.append(pd.Series(labelnum),ignore_index=True)
        print("stopped")
        break
    # same label
    if labelname in labellist:
        index=labellist.index(labelname)
    # new label
    else:
        labellist.append(labelname)
        index=len(labellist)-1

    # how many
    print("how many? (enter 0 to exit)")
    try:
        sets=int(input())
    except:
        break
    if sets==0:
        data=data.append(pd.Series(labelnum),ignore_index=True)
        dataraw=dataraw.append(pd.Series(labelnum),ignore_index=True)
        dataref=dataref.append(pd.Series(labelnum),ignore_index=True)
        print('stopped')
        break

    # measuring reference
    print("measuring reference")
    check=con.refstart()#reference
    if check==0:
        print("reference measured.")
        while True:
            print("Ready? (y)")
            ready = input()
            if ready == "y":
                print("Starting measurement")
                break
        print("start")
    if check!=0:
        print("Data could not be obtained. Stopping measurement")
        break

    # measureing data
    plt.ion()
    for i in range(sets):
        con.oneline()
        ##### change vec type here
        tmp=con.getvec() #rawdata-ref and ifft (3*FREQ_SIZE->3072)
        tmpraw=con.getrawvec() #rawdata (2*FREQ_SIZE->2048)
        tmpref=con.getrefvec() #reference (2*FREQ_SIZE->2048)
        data[rows]=tmp ##### also change here
        dataraw[rows]=tmpraw
        dataref[rows]=tmpref

        # need nparray (1,10240) for predction
        darray = data.query('10239 < index < 20480')[rows].values # converting into array
        print("darray.shape", darray.shape)
        reshaped = np.reshape(darray, (1, darray.shape[0]))
        print("reshaped.shape", reshaped.shape)
        predictions = new_model.predict([reshaped])
        print("Prediction array:{}, argmax:{}".format(predictions[0,:], np.argmax(predictions[0])))

        print('label {} No.{}'.format(labelname, i))
        xlabel = 'Data No.{}'.format(i) # 直接xlabelの中で書いたらエラーがでた
        plt.xlabel(xlabel)
        plt.ylim(-Yscope,Yscope) # y axis limitation
        plt.plot(reshaped[0]) # plotting array


        # print(data.query('10240 < index < 20480')[rows].shape)
        # plt.plot(dataraw[rows])
        # plt.plot(dataref[rows])
        # plt.show()
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        rows+=1
    plt.close() #

    # combining aquiried label and exsisting label by np.hstack
    labelnum=np.hstack((labelnum,index*np.ones(sets)))
dataT=data.T
datarawT=dataraw.T
datarefT=dataref.T

# getting label name summary for filename
i=0
labelnames = ""
while i <= index:
    labelnames = labelnames + "-" + labellist[i]
    i = i+1

# saving data
while True:
    print("Save? (y/n)")
    saving = input()
    if saving == "y":
        print("Saving ...")
        data.to_csv('{}{}.csv'.format(time.strftime('%Y-%m-%d-%H-%M-%S'),labelnames))
        print("Saved")
        break
    if saving == "n":
        print("Data was not saved")
        break

print("DONE")
