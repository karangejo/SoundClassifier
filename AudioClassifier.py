import librosa
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import matplotlib.pyplot as plt

# directory where all the files and datasets are stored
home_dir = '/home/karang/Documents/AudioClassification/'

# helper function to get the category label from the folder containing the samples
def get_category_from_path_string(pathstring):
    splits = pathstring.split('/')
    length = len(splits)
    return(splits[length-1])

# this function calculates the stft and mfccs from a .wav file
def get_features(filepath):
    # load the file
    X, sample_rate = librosa.load(filepath)
    # calculate the features
    stft = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    # clean them up a bit
    stftM = np.mean(stft.T, axis=0)
    mfccsM = np.mean(mfccs.T, axis=0)
    # return everything
    return(stftM,mfccsM)

# this function gets a dataset with stft and mfccs data
# from a directory tree of subfolders of .wav files
# the folder names are the category labels for the data
def get_dataset_from_folders(rootdir):
    # empty list to store the rows of data
    df = []
    # go through the directory tree to get and label the data
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            # get the features
            stft,mfccs = get_features(root+'/'+file)
            # get the label
            label = get_category_from_path_string(root)
            # append it to our list
            df.append([label,stft,mfccs])
    # convert to pandas dataframe
    complete_df = pd.DataFrame(df, columns=['label','stft','mfccs'])
    return(complete_df)

# this function returns a list with our prepared training and test data
# separated into X and Y components
def prepare_data_stft(df):
    # prepare the X and Y components
    X = np.array(df.stft.tolist())
    y = np.array(df.label.tolist())
    # encode the classification labels
    le = LabelEncoder()
    Y = to_categorical(le.fit_transform(y))
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.1, random_state=1)
    #return it
    return(x_train,x_test,y_train,y_test,le)

# this function returns a list with our prepared training and test data
# separated into X and Y components
def prepare_data_mfccs(df):
    # prepare the X and Y components
    X = np.array(df.mfccs.tolist())
    y = np.array(df.label.tolist())
    # encode the classification labels
    le = LabelEncoder()
    Y = to_categorical(le.fit_transform(y))
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.1, random_state=1)
    #return it
    return(x_train,x_test,y_train,y_test,le)

# this function builds the neural net for us
# and prints some initial information
def build_neural_net(x_trn, x_tst, y_trn, y_tst):

    num_labels = y_trn.shape[1]

    # build model
    model = Sequential()

    model.add(Dense(256, input_shape=x_trn[0].shape))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # Display model architecture summary
    model.summary()

    # Calculate pre-training accuracy
    score = model.evaluate(x_tst, y_tst, verbose=1)
    accuracy = 100*score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)

    return(model)

# this function actually trains the neural net
# backs it up to a file and evaluates it on the training and test data
def train_neural_net(model,num_epochs,num_batch_size,x_trn, x_tst, y_trn, y_tst):
    # save a checkpoint in case something goes wrong while training
    checkpointer = ModelCheckpoint(filepath=home_dir + 'checkpointModel.hdf5',verbose=1,save_best_only=True)

    start = datetime.now()
    # fit the model
    history = model.fit(x_trn,y_trn, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_tst,y_tst), callbacks=[checkpointer], verbose=1)

    duration = datetime.now() - start

    # print some info
    print("Training completed in time: ", duration)

    score = model.evaluate(x_trn,y_trn,verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_tst,y_tst,verbose=0)
    print("Testing Accuracy: ", score[1])

    return(model,history)

def plot_training(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# save model to disk
def save_model(model):
    model.save(home_dir + 'savedModel.h5')
    print("Saved model to disk")

# load model and return it
def load_model_from_disk():
    model = load_model(home_dir + 'savedModel.h5')
    model.summary()
    print("Model loaded and returned")
    return(model)

def get_prediction_stft(audio_file_path,model,le):
    stft, mfccs = get_features(audio_file_path)
    result = model.predict(np.array([stft]))
    predictions = [np.argmax(y) for y in result]
    label = le.inverse_transform(predictions)[0]
    return(label)

def get_prediction_mfccs(audio_file_path,model,le):
    stft, mfccs = get_features(audio_file_path)
    result = model.predict(np.array([mfccs]))
    predictions = [np.argmax(y) for y in result]
    label = le.inverse_transform(predictions)[0]
    return(label)

def run_stft_model(epochs, batch_size):
    df = get_dataset_from_folders(home_dir +'dataset')
    xtrain, xtest, ytrain, ytest, le = prepare_data_stft(df)
    model = build_neural_net(xtrain, xtest, ytrain, ytest)
    model, history = train_neural_net(model,epochs,batch_size,xtrain, xtest, ytrain, ytest)
    plot_training(history)
    save_model(model)
    return(model,history,le)

def run_mfccs_model(epochs, batch_size):
    df = get_dataset_from_folders(home_dir +'dataset')
    xtrain, xtest, ytrain, ytest, le = prepare_data_mfccs(df)
    model = build_neural_net(xtrain, xtest, ytrain, ytest)
    model, history = train_neural_net(model,epochs,batch_size,xtrain, xtest, ytrain, ytest)
    plot_training(history)
    save_model(model)
    return(model,history,le)
