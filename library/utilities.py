import scipy.io as sio
import numpy as np
import os
import glob

from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, learning_curve

import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, neighbors, datasets
from sklearn import svm
import scipy.signal as signal
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import LeavePOut
from sklearn.preprocessing import StandardScaler
import sys
import time
import os
import math

import shutil
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import pickle as pk
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import keras as ke
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, UpSampling2D,Conv2DTranspose,ZeroPadding2D,MaxPooling1D,MaxPooling2D, CuDNNLSTM, Dropout, Reshape, PReLU, ELU, BatchNormalization, Flatten
from keras import optimizers
from keras import initializers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.utils import to_categorical





def get_patients():
    patients_path = glob.glob('./data/standard/p*')
    patients = []
    for patient in sorted(patients_path):
        patients.append(patient[-2:])
    return list(set(patients))


def get_visits(patient):
    visit_path = glob.glob('./data/standard/p_' + patient + '/v_*')
    visits = []
    # print(visit_path)
    for visit in sorted(visit_path):
        visits.append(visit[-2:])
    return list(set(visits))


def get_days(visit, patient):
    day_path = glob.glob('./data/standard/p_' + patient + '/v_' + visit + '/Features/F*')
    days = []
    for day in sorted(day_path):
        days.append(day[-8])
    return sorted(list(set(days)))


def block_performances(day, visit, patient, pipeline, x_full=None, y_full=None, df=None, block_size=20):
    if df is None:
        df = pd.DataFrame()  # Create a new dataframe, if not append to it
    if x_full is None:
        x_full, y_full = load_data(day=day, visit=visit, patient=patient, whole=True)
    for n_questions in range(block_size, x_full.shape[0] + 1 - block_size, block_size):
        x_train = x_full[:n_questions, :]
        x_test = x_full[n_questions:, :]
        y_train = y_full[:n_questions]
        y_test = y_full[n_questions:]
        pipeline.fit(x_train, y_train)
        df = df.append({'Day': day,
                        'Train size': n_questions,
                        'Test size': x_full.shape[0] - n_questions,
                        'Score on train': pipeline.score(x_train, y_train),
                        'Score on test': pipeline.score(x_test, y_test),
                        'visit': visit,
                        'patient': patient,
                        },
                       ignore_index=True)
    return df


def plot_block_perf(df, title):
    sns.set_style('whitegrid', {'xtick.bottom': True, 'ytick.left': True})
    sns.lineplot(x='Train size',
                y='Score on test',
              #  kind='line',
                # col='visit',
                # row='patient',
                hue='pipeline',
                style='pipeline',
                markers=True, dashes=False,
                data=df)
    plt.ylim((0, 1.05))
    plt.plot([0, 70], [0.7, 0.7], 'r:', linewidth=1.5)
    plt.yticks(np.arange(0, 1.05, step=0.1))
    plt.title(title)
    plt.show()


def load_data(patient, visit, day, n_questions=20, whole=False, dimensionality_reduction=-1, reduction_neighbors=15,
              reduction_metric='hamming', scaler=None):
    '''
    :param
        day: the day you want to load
        n_questions: if int, the number of question (in temporal order) that you want to keep as train.
                    if between 0 and 1 represent the fraction of dataset you want to keep in train
        whole: boolean, if True return the whole dataset.

    :return: the labeled dataset for the selected day already partitioned in train/test
    '''
    # Import data from mat files in two dictionaries
    data_path_unordered = glob.glob(
        './data/standard/p_' + patient + '/v_' + visit + '/Features/Features_p' + patient + '_v' + visit + '_d' + day + '*')
    file_list = sorted(data_path_unordered)
    # unordered_file_list = glob.glob(folder + 'Features_d' + str(day) + '*')
    # file_list = sorted(unordered_file_list)  # sorted to keep the actual temporal order of the blocks
    data = {}
    labels = {}
    print(file_list)
    # print('./data/patient_' + patient + '/unpro_visit_' + visit + '/FeaturesUnprocessedData_p11_v01_d' + day + '*')
    for file in file_list:
        mat_dict = sio.loadmat(file)
        # data[file] = mat_dict['features']
        data[file] = mat_dict['features']
        labels[file] = mat_dict['answerLabels']

    # From the dictionaries create two matrix, one of features one of labels
    full_matrix = data[file_list[0]]
    full_labels = labels[file_list[0]]
    for day in data.keys():
        if day != file_list[0] and labels[day].shape[0] > 0:  # 2nd condition for datasets that contain open questions
            features_block = data[day]
            labels_block = labels[day]
            full_matrix = np.concatenate((full_matrix, features_block))
            full_labels = np.concatenate((full_labels, labels_block))
    full_labels = full_labels.flatten()  # flatten the vector

    if whole:
        return full_matrix, full_labels

    # Split in train/test
    if n_questions > full_matrix.shape[0] or n_questions == 0:
        raise ValueError
    elif n_questions >= 1:
        x_train = full_matrix[:n_questions, :]
        x_test = full_matrix[n_questions:, :]
        y_train = full_labels[:n_questions]
        y_test = full_labels[n_questions:]
    elif n_questions < 1:  # act as if it were the % kept in train
        n_questions = int(n_questions * full_matrix.shape[0])
        x_train = full_matrix[:n_questions, :]
        x_test = full_matrix[n_questions:, :]
        y_train = full_labels[:n_questions]
        y_test = full_labels[n_questions:]

    if dimensionality_reduction > 0:
        scaler = RobustScaler()
        x_train = scaler.fit_transform(x_train, y_train)
        x_test = scaler.transform(x_test)
        reducer = umap.UMAP(n_neighbors=reduction_neighbors, n_components=dimensionality_reduction,
                            metric=reduction_metric)
        x_train = reducer.fit_transform(x_train, y_train)
        x_test = reducer.transform(x_test)

    if scaler is not None:
        if scaler == 'standard':
            scl = StandardScaler()
            x_train = scl.fit_transform(x_train, y_train)
            x_test = scl.transform(x_test)
        elif scaler == 'robust':
            scl = RobustScaler()
            x_train = scl.fit_transform(x_train, y_train)
            x_test = scl.transform(x_test)

    return x_train, x_test, y_train, y_test


def iterative_score(clf, day, question_to_start=20):
    x, y = load_data(day=day, whole=True)
    scores = []
    for idx in range(question_to_start, x.shape[0]):
        # scaler = StandardScaler()
        x_train = x[:idx, :]
        x_test = x[idx, :].reshape(1, -1)
        y_train = y[:idx]
        y_test = [y[idx]]
        weight = np.linspace(1, 2, x_train.shape[0])
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))
    print(scores)
    return np.mean(scores), np.std(scores)


def plot_temporal_score(x_full, y_full, pipeline, title_label, old=False, order=None):
    df = pd.DataFrame()

    for n in range(10, x_full.shape[0] - 1):
        if old:
            x_train, x_test, y_train, y_test = get_experimental_sequence(x_full, y_full, order, n)
        else:
            x_train = x_full[:n, :]
            y_train = y_full[:n]
            x_test = x_full[n:, :]
            y_test = y_full[n:]

        pipeline.fit(x_train, y_train)
        sc = pipeline.score(x_test, y_test) * 100

        print({'train_size': n, 'test_size': x_full.shape[0] - n, 'score': sc})
        df = df.append({'train_size': n, 'test_size': x_full.shape[0] - n, 'score': sc}, ignore_index=True)
    sns.set()
    sns.lineplot(x='train_size', y='score', data=df)
    plt.ylim((0, 105))
    plt.title(title_label)
    plt.xlabel('Train set size')
    plt.ylabel('Accuracy score %')
    plt.show()


def plot_learning_curve(x_full, y_full, pipeline, title, num_cv_fold=100, train_size=0.9, show=True, save=False,
                        save_title='fig', n_jobs=4, verbose=0):
    cv = ShuffleSplit(n_splits=num_cv_fold, train_size=train_size)
    train_sizes, train_scores, test_scores = learning_curve(pipeline, x_full, y_full, cv=cv,
                                                            n_jobs=n_jobs, verbose=verbose,
                                                            train_sizes=np.linspace(10 / x_full.shape[0], 1,
                                                                                    int(x_full.shape[
                                                                                            0] * train_size - 10)))
    print(test_scores.shape)
    df = pd.DataFrame()
    for idx in range(len(train_sizes)):
        for fold in range(num_cv_fold):
            df = df.append({'train_size': train_sizes[idx], 'train_score': train_scores[idx, fold] * 100,
                            'test_score': test_scores[idx, fold] * 100}, ignore_index=True)

    sns.set()
    sns.lineplot(x='train_size', y='train_score', data=df)
    sns.lineplot(x='train_size', y='test_score', data=df)
    plt.ylim((0, 105))
    plt.title(title)
    plt.xlabel('Train instances considered')
    plt.ylabel('Accuracy score %')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc=0)
    if save:
        plt.savefig('figures/' + save_title, format='pdf')
    if show:
        plt.show()


def channels_to_vector(channels):
    time_instances = [];
    dim = channels.shape;
    # find the length min of the signal in the specified temporal instance
    length_min = len(channels[0, 1]);
    for i in range(0, dim[1]):
        single_measurement = channels[0, i];
        single_length = single_measurement.shape[0]
        if (single_length < length_min):
            length_min = single_length;
    # export the signals
    for i in range(0, dim[1]):
        single_measurement = channels[0, i];
        dim1 = single_measurement.shape;
        time_instance = [];
        for j in range(0, dim1[1]):
            if (len(single_measurement[:, j]) > length_min):
                single_signal = single_measurement[:, j][0:length_min]
            else:
                single_signal = single_measurement[:, j]
            # put in a list
            time_instance.append(np.asarray(single_signal).reshape(len(single_signal), 1).T);
        # create the matrix of the signals per a single time instance
        time_instance = np.concatenate(time_instance);
        time_instances.append(time_instance);
    return time_instances;


##
# Create the train data matrix
##
## usage
def get_feature_matrix_and_labels(channel_structure, label, features_extracted, connectivity_feature):
    list_train = []
    list_labels = []
    cont = 0;
    index_connectivity = 0;
    list_row = []

    for time_instance in channel_structure:
        dim1 = time_instance.shape
        # indipendent_components=extract_ICs(time_instance,n_ICA_components);
        for j in range(0, dim1[0]):
            features = features_extracted[cont, :];
            list_row.append(features);
            cont = cont + 1;
        list_row.append(connectivity_feature[index_connectivity, :]);
        index_connectivity = index_connectivity + 1;
        labels = get_labels(1, label);
        feature_row = np.concatenate(list_row);
        list_train.append(feature_row.reshape(len(feature_row), 1).T)
        list_labels.append(labels);
        list_row = []

    train_TX = np.concatenate(list_train)
    labels = np.concatenate(list_labels, axis=0)

    return train_TX, labels.T.reshape(labels.size)


def get_feature_matrix_and_labels_new(channel_structure, label, features_extracted):
    list_train = []
    list_labels = []
    cont = 0;
    index_connectivity = 0;
    list_row = []

    for time_instance in channel_structure:
        dim1 = time_instance.shape
        # indipendent_components=extract_ICs(time_instance,n_ICA_components);
        for j in range(0, dim1[0]):
            features = features_extracted[cont, :];
            list_row.append(features);
            cont = cont + 1;
        # list_row.append(connectivity_feature[index_connectivity,:]);
        # index_connectivity=index_connectivity+1;
        labels = get_labels(1, label);
        feature_row = np.concatenate(list_row);
        list_train.append(feature_row.reshape(len(feature_row), 1).T)
        list_labels.append(labels);
        list_row = []

    train_TX = np.concatenate(list_train)
    labels = np.concatenate(list_labels, axis=0)

    return train_TX, labels.T.reshape(labels.size)


### Description
def get_labels(number, string):
    if (string == "No"):
        return np.zeros(number)
    if (string == "Yes"):
        return np.ones(number)


def get_experimental_sequence(X, y, order, c):
    idx_tr = []
    y_tr = []
    half = int(X.shape[0] / 2)
    # c = number of minimum instances
    i = 0
    j = 0
    for element in order:
        if have_two_classes(y_tr) and len(y_tr) >= c:
            break
        if element == 1:  # 1 is YES
            idx_tr.append(i)
            # X_tr.append(X[i,:])
            y_tr.append(y[i])
            i += 1
        else:  # 0 is NO
            idx_tr.append(j + half)
            # X_tr.append(X[half+j,:])
            y_tr.append(y[half + j])
            j += 1

    X_tr = X[idx_tr, :]
    y_tr = y[idx_tr]

    idx_te = [k for k in range(X.shape[0])]
    for l in idx_tr:
        idx_te.remove(l)
    X_te = X[idx_te, :]
    y_te = y[idx_te]

    return X_tr, X_te, y_tr, y_te


def have_two_classes(y):
    have_zero = False
    have_one = False
    for element in y:
        if element == 1: have_zero = True
    for element in y:
        if element == 0: have_one = True
    return have_zero & have_one


def extract_dataset(destination_folder):
    # Import data from mat files
    old_path = os.getcwd()
    os.chdir(destination_folder)
    yes_EEG_contents = sio.loadmat('EEGyes.mat')
    no_EEG_contents = sio.loadmat('EEGno.mat')

    channels_no_EEG = no_EEG_contents["EEGno"]
    channels_yes_EEG = yes_EEG_contents["EEGyes"]

    # Features Loading
    features_extracted_yes = sio.loadmat('FeaturesYes.mat')['FeaturesYes']
    features_extracted_no = sio.loadmat('FeaturesNo.mat')['FeaturesNo']
    connectivity_feature_yes = sio.loadmat('ConnectivityFeaturesYes.mat')['ConnectivityFeaturesYes']
    connectivity_feature_no = sio.loadmat('ConnectivityFeaturesNo.mat')['ConnectivityFeaturesNo']

    channels_structure_yes_EEG = channels_to_vector(channels_yes_EEG)
    channels_structure_no_EEG = channels_to_vector(channels_no_EEG)

    ##Structuring of the data:
    # the code below create the train matrix with respect to the signal given in "channel_structure" but using the features contained in "features_extracted*" and in "connettivity_feature*".
    feature_dataset_yes_EEG, EEG_yes_labels = get_feature_matrix_and_labels(channels_structure_yes_EEG, "Yes",
                                                                            features_extracted_yes,
                                                                            connectivity_feature_yes);

    feature_dataset_no_EEG, EEG_no_labels = get_feature_matrix_and_labels(channels_structure_no_EEG, "No",
                                                                          features_extracted_no,
                                                                          connectivity_feature_no);

    # Merge the labeled data
    feature_dataset_full = np.concatenate((feature_dataset_yes_EEG, feature_dataset_no_EEG), axis=0)
    labels = np.concatenate((EEG_yes_labels, EEG_no_labels), axis=0)

    # Order
    order = sio.loadmat('order.mat')['order']
    order = order.flatten()

    os.chdir(old_path)
    return feature_dataset_full, labels, order


def extract_dataset_new(destination_folder):
    # Import data from mat files
    old_path = os.getcwd()
    os.chdir(destination_folder)
    yes_EEG_contents = sio.loadmat('EEGyes.mat')
    no_EEG_contents = sio.loadmat('EEGno.mat')

    channels_no_EEG = no_EEG_contents["EEGno"]
    channels_yes_EEG = yes_EEG_contents["EEGyes"]

    # Features Loading
    features_extracted_yes = sio.loadmat('FeaturesYes.mat')['FeaturesYes']
    features_extracted_no = sio.loadmat('FeaturesNo.mat')['FeaturesNo']
    # connectivity_feature_yes = sio.loadmat('ConnectivityFeaturesYes.mat')['ConnectivityFeaturesYes']
    # connectivity_feature_no  = sio.loadmat('ConnectivityFeaturesNo.mat')['ConnectivityFeaturesNo']

    channels_structure_yes_EEG = channels_to_vector(channels_yes_EEG)
    channels_structure_no_EEG = channels_to_vector(channels_no_EEG)

    ##Structuring of the data:
    # the code below create the train matrix with respect to the signal given in "channel_structure" but using the features contained in "features_extracted*" and in "connettivity_feature*".
    # feature_dataset_yes_EEG, EEG_yes_labels = get_feature_matrix_and_labels_new(channels_structure_yes_EEG,"Yes",features_extracted_yes);

    # feature_dataset_no_EEG, EEG_no_labels = get_feature_matrix_and_labels_new(channels_structure_no_EEG,"No",features_extracted_no);

    EEG_yes_labels = np.ones(features_extracted_yes.shape[0])
    EEG_no_labels = np.zeros(features_extracted_no.shape[0])

    # Merge the labeled data
    feature_dataset_full = np.concatenate((features_extracted_yes, features_extracted_no), axis=0)
    labels = np.concatenate((EEG_yes_labels, EEG_no_labels), axis=0)

    # Order
    order = sio.loadmat('order.mat')['order']
    order = order.flatten()

    os.chdir(old_path)

    return feature_dataset_full, labels, order


def import_data(list_folders, directory, new=False):
    data = {}
    for folder_name in list_folders:
        folder_path = directory + folder_name
        if new:
            data[folder_name] = extract_dataset_new(folder_path)
        else:
            data[folder_name] = extract_dataset(folder_path)

    return data







def split_matrix_two_blocks(y, percentage1, percentage2, seed):
    """Build k indices for k-fold."""
    if(percentage1+percentage2==1):
        num_row = len(y)
        #print(num_row)
        interval_1 = int(percentage1*num_row);
        
        np.random.seed(seed)
        indices = np.random.permutation(num_row);
        first_indices = indices[0:interval_1];
        second_indices = indices[interval_1:num_row];
        return [np.array(first_indices),np.array(second_indices)]
    else:
        print('>>>>>>>>>>>ERROR:Not valid splitting percentage')

def slidingWindow(sequence, labels, winSize, step):

    # Verify the inputs

    try: it = iter(sequence)

    except TypeError:

        raise Exception("**ERROR** sequence must be iterable.")

    if not ((type(winSize) == type(0)) and (type(step) == type(0))):

        raise Exception("**ERROR** type(winSize) and type(step) must be int.")

    if step > winSize:

        raise Exception("**ERROR** step must not be larger than winSize.")

    if winSize > len(sequence):

        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # number of chunks
    numOfChunks = ((len(sequence)-winSize)//step)+1
    segment=[]
    seg_labels=[]
    # Do the work
    for i in range(0,numOfChunks*step,step):
        segment.append(sequence[i:i+winSize])
        seg_labels.append(labels[i:i+winSize])
    return segment,seg_labels



def slidingWindow_autoencoder(sequence, winSize, step):

    # Verify the inputs

    try: it = iter(sequence)

    except TypeError:

        raise Exception("**ERROR** sequence must be iterable.")

    if not ((type(winSize) == type(0)) and (type(step) == type(0))):

        raise Exception("**ERROR** type(winSize) and type(step) must be int.")

    if step > winSize:

        raise Exception("**ERROR** step must not be larger than winSize.")

    if winSize > len(sequence):

        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # number of chunks
    numOfChunks = ((len(sequence)-winSize)//step)+1
    segment=[]
    
    # Do the work
    for i in range(0,numOfChunks*step,step):
        segment.append(sequence[i:i+winSize])

                
    return np.array(segment)



            
            
            
def segment_data(X_train, labels, winSize, step):
        # obtain chunks of data

        train_segments,labels = slidingWindow(X_train, labels,winSize, step)
        train_labels = []
        
        for single_labels in labels:
            train_labels.append(get_most_frequent(single_labels))
        
        return np.array(train_segments), np.array(train_labels)
    
    
def prepare_data(train_data):
    #encoder = OneHotEncoder()
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    return train_data;


def reshaping(data):
    return data.reshape(data.shape[0],data.shape[2],data.shape[1])


def get_most_frequent(labels):

    (values, counts) = np.unique(labels, return_counts=True)
    index = np.argmax(counts)
    return values[index]



def get_accuracy(predicted_labels, true_labels):
     if (predicted_labels.size == true_labels.size):
        return  np.sum(predicted_labels ==  true_labels )/len( true_labels)

def classification_SVM_experiments_std(X, Y, classifier, fraction_train_test, num_experiments,Model_Name,save_model):
    
    seed=range(num_experiments)
    svm_total_acc_test  = []
    svm_total_acc_train = [] 
    dataset_length=X.shape[0];
    
    for single_seed in seed:
        scaler = StandardScaler()
        
        [i1,i2]=split_matrix_two_blocks(X, fraction_train_test, 1-fraction_train_test,single_seed)
        
        
        train =X[i1,:]
        labels_train=Y[i1]
        
        test = X[i2,:]
        labels_test=Y[i2]
        

  
        clf=classifier
        clf.fit(train, labels_train)  

        
        #Accuracy on test
        predicted_labels_test = clf.predict(test)
        SVM_accuracy_test = get_accuracy(predicted_labels_test, labels_test)
        svm_total_acc_test.append(SVM_accuracy_test)
        
        """if(save_model==1):
            old_path=os.getcwd()
            os.chdir('Model'+Model_Name)
            joblib.dump(clf, 'Classifier_'+Model_Name+'.pkl') 
            os.chdir(old_path)"""
    
        #Accuracy on train
        predicted_labels_train = clf.predict(train)
        SVM_accuracy_train = get_accuracy(predicted_labels_train, labels_train)
        svm_total_acc_train.append(SVM_accuracy_train)
        #print("Accuracy: "+ str(SVM_accuracy) + "; iteration  " + str(single_seed) )
    return svm_total_acc_test, svm_total_acc_train




def performance_assesment_fraction_std(X, Y, num_experiment, classifier,Model_Name):
    fracs = np.linspace(0.3,0.9,20)
    accuracy_test_mean  = []
    accuracy_test_std   = []
    accuracy_train_mean = []
    accuracy_train_std  = []

    for frac_tr_te in fracs:
        print("Evaluation progress: " + str(int((frac_tr_te-fracs[0])/(fracs[-1]-fracs[0])*100)) + " %")
        acc_test, acc_train = classification_SVM_experiments_std(X, Y, classifier, frac_tr_te, num_experiment,Model_Name,0)
        #saving of metrics of interest
        accuracy_test_mean.append(np.mean(acc_test))
        accuracy_test_std.append(np.std(acc_test))
        accuracy_train_mean.append(np.mean(acc_train))
        accuracy_train_std.append(np.std(acc_train))
    
    frac_tr_te=0.95;
    acc_test, acc_train =classification_SVM_experiments_std(X, Y, classifier, frac_tr_te, 1,Model_Name,1)
    #plot the figure
    plt.figure(figsize=(10, 7), dpi=80)
    plt.errorbar(fracs, accuracy_test_mean, yerr=accuracy_test_std, label="Error bars plot", fmt="s-",  linewidth=3)
    plt.errorbar(fracs, accuracy_train_mean, yerr=accuracy_train_std, label="Error bars plot", fmt="s-",  linewidth=3)
    plt.grid(b=True, which='major', color='k', linestyle='--', alpha = 0.4)
    plt.minorticks_on()
    plt.title('SVM perfomances over different train/test dataset of reduced features')
    plt.ylabel('Accuracy')
    plt.xlabel('Train instances considered')

    plt.legend(['Test Accuracy', 'Train Accuracy'], loc=4)
    name='train_test_acc_fine_tuned_IID_'+Model_Name+'.eps'
    #plt.savefig(name, format='eps')
    plt.show()
    
    
def get_data(name,num_channels,window_size,step_size):
    
    data_matrix=[]
    data_matrix_1=[]
    labels_1=[]
    labels=[]
    list_files=os.listdir(name)

    train_data=[]
    block_dimension=[]
    for single_file in list_files:
        if(single_file.startswith( 'Signal' )==1):
            print(single_file)
            data = sio.loadmat(name+'/'+single_file)
            signal=reshaping(data['signal']);
            signal=select_channels(signal,num_channels)
            label=data['labels']
            data_matrix.append(signal)
            block_dimension.append(signal.shape)
            labels.append(label)

    data_concat=np.concatenate(data_matrix)
    label_concat=np.concatenate(labels)
    temp_label=[];

    # reshape labels

    for i in range(0,label_concat.shape[0]):
        segmented_labels=np.ones((data_concat.shape[1],1))*label_concat[i];
        temp_label.append(segmented_labels)
        

    #concatenate matrix 
    label_concat=np.concatenate(temp_label)
    data_concat=data_concat.reshape(data_concat.shape[0]*data_concat.shape[1],data_concat.shape[2])
    print(data_concat.shape)
    data_concat=prepare_data(data_concat)
    
   # create matrix blocks 
    blocked_data,blocked_labels=get_blocked_data(data_concat,label_concat,block_dimension)
   
    labels=[]
    data_matrix=[]
    for single_block,single_labels in zip(blocked_data,blocked_labels):
    
        segmented_data,segmented_label=segment_data(single_block,single_labels,window_size,step_size);
        data_matrix.append(segmented_data);
        labels.append(segmented_label)
        
        
    
    labels=np.concatenate(labels)
    data_matrix=np.concatenate(data_matrix)
    
    print('Data ready.. ')
    
    print('Signal samples of shapes: ' +  str(data_matrix.shape))
    
    print('Labels of shapes: ' +  str(len(labels)) )
    return data_matrix,labels
def get_blocked_data(data_concat,labels,block_dimension):
    block=[]
    block_labels=[]
    last_dimension=0;

    for single_dimension in block_dimension:
        for i in range(0,single_dimension[0]):
            block.append(data_concat[range(last_dimension,last_dimension+single_dimension[1]),:])
            block_labels.append(labels[range(last_dimension,last_dimension+single_dimension[1]),:])
            last_dimension=last_dimension+single_dimension[1];
    return block,block_labels
    
        
        
        
def get_autoencoder_data(path, name,num_channels,window_size,step_size):
    
    train_data=[]
    for single_name in name:
    
        list_files=os.listdir(path+single_name)
        for single_file in list_files:
            if(single_file.startswith('Auto')==1):
                print(single_file)
                data = sio.loadmat(path+single_name+'/'+single_file)
                highpass=data['highfiltered'].T
                train_data.append(highpass[:,range(0,num_channels)])

    train_data=np.concatenate(train_data);
    train_data=prepare_data(train_data);
    train_segments = slidingWindow_autoencoder(train_data, int(window_size), int(step_size))
    reshaped_train = train_segments.reshape(train_segments.shape[0], int(window_size), num_channels, 1)
    
    print('Data for Autoencoder ready.. ')
    
    print('Signal samples of shapes: ' +  str(reshaped_train.shape))
    
    
    return reshaped_train


def select_channels(data,num_channels):
    return data[:,:,range(0,num_channels)]

