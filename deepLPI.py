import keras
from keras.layers import Input, LSTM, Dense, Dropout, Activation, Flatten
from keras.models import Model
import numpy as np
from keras.utils import to_categorical
from numpy import array
from keras import regularizers
import numpy as np
import pandas as pd
import csv
import scipy.io as sio
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D
import scipy.io as sio
import ast
import sys
import time
from random import shuffle
import numpy as np
import argparse
import unicodedata
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Input, Dense, Layer, Dropout, Reshape
from mil_nets.dataset import load_dataset
from mil_nets.layer import Score_pooling
from mil_nets.metrics import bag_accuracy
from mil_nets.objectives import bag_loss
from mil_nets.utils import convertToBatch
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras import activations, initializers, regularizers
from keras.layers import Layer
import random
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, SpatialDropout1D
from keras.layers import Activation, Flatten, Input, Masking
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import regularizers
from keras import losses
from keras import optimizers
from keras.models import Model
from crf import CRF
#from keras.utils import multi_gpu_model
#from utils import generateLabel, upSample, makeBatch
import time
from sys import argv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

"""
Import Error: undefined symbol: PyUnicodeUCS4_AsASCIIString
Python is build with UCS2 ascii code

To determine if your python installation is UCS2 or UCS4, from a python shell type:
 
#When built with --enable-unicode=ucs4:
#>>> import sys 
#>>> print sys.maxunicode 
#1114111

#When built with --enable-unicode=ucs2:
#>>> import sys 
#>>> print sys.maxunicode 
#65535

Try using: Python/2.7.5
"""

import vis
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

def plot_RNA_data(data, filename):
    y = data
    x = np.arange(len(data))
    plt.bar(x,y)
    plt.rcParams['figure.figsize'] = (18, 6)
    plt.savefig(filename)

def visulize_saliency(model, lncRNA_b, mRNA_b, label_b):
    class_idx = 0
    indices = np.where(label_b[:] == 1)[0]
    plot_RNA_data(lncRNA_b[0], 'lncRNA_sample.pdf')
    plot_RNA_data(mRNA_b[0], 'mRNA_sample.pdf')
    layer_idx = utils.find_layer_idx(model, 'score_pooling')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    

    for modifier in ['guided', 'relu']:
        model.summary()
        grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=lncRNA_b, backprop_modifier=modifier)
        plt.figure()
        plt.title(modifier)
        plt.plot(grads)
        plt.savefig('test.pdf')


def precision_cal(y_list, p_list):
    y = np.array(y_list)
    pred = np.array(p_list)
    p, r, thresholds = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(r, p)
    return auprc


def get_expr_data(filename):
    mRNA_exp_dict={}
    mRNA_reader = csv.reader(open(filename, "r"))
    print(mRNA_reader)
    for mRNAs in mRNA_reader:
        mRNA_exp_dict[mRNAs[0].split('.')[0]] = [float(mRNAs[1])]
        for i in range(len(mRNAs)-2):
            mRNA_exp_dict[mRNAs[0].split('.')[0]].append(float(mRNAs[i+2]))
    return mRNA_exp_dict



def model_func(lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len):

    tweet_a = Input(shape=(lncRNA_len,)) #50, 4))
    encoded_a1 = Embedding(input_dim=4096, output_dim=64)(tweet_a)
    encoded_a1 = Convolution1D(filters = 4, kernel_size = 32, strides = 1,  padding = 'valid', activation = 'relu')(encoded_a1)
    encoded_a1 = MaxPooling1D(pool_size=4, strides=None, padding='valid')(encoded_a1)
    encoded_a1 = Dropout(0.2)(encoded_a1)
    encoded_a1 = LSTM(32, return_sequences=True)(encoded_a1)

    tweet_b = Input(shape=(mRNA_len,))
    encoded_b1 = Embedding(input_dim=8000, output_dim=64)(tweet_b)
    encoded_b1 = Reshape((lncRNA_len,64*(mRNA_len/lncRNA_len)))(encoded_b1)
    encoded_b1 = Convolution1D(filters = 4, kernel_size = 32, strides = 1,  padding = 'valid', activation = 'relu')(encoded_b1)
    encoded_b1 = MaxPooling1D(pool_size=4, strides=None, padding='valid')(encoded_b1)
    encoded_b1 = Dropout(0.2)(encoded_b1)
    encoded_b1 = LSTM(32, return_sequences=True)(encoded_b1)
    
    tweet_c = Input(shape=(lncRNA_struct_len,6))
    encoded_c1 = Convolution1D(4, kernel_size = 32, strides = 1, padding = 'valid', activation = 'relu')(tweet_c)
    encoded_c1 = MaxPooling1D(pool_size=4, strides=None, padding='valid')(encoded_c1)
    encoded_c1 = Dropout(0.2)(encoded_c1)
    encoded_c1 = LSTM(32, return_sequences=True)(encoded_c1)
    
    tweet_d = Input(shape=(mRNA_struct_len,6))
    encoded_d1 = Convolution1D(4, kernel_size = 32, strides = 1, padding = 'valid', activation = 'relu')(tweet_d)
    encoded_d1 = MaxPooling1D(pool_size=4, strides=None, padding='valid')(encoded_d1)
    encoded_d1 = Dropout(0.2)(encoded_d1)
    encoded_d1 = LSTM(32, return_sequences=True)(encoded_d1)

    merged_vector_lnc = keras.layers.concatenate([encoded_a1, encoded_c1], axis=-1, name='merged_vector_lnc')
    merged_vector_m = keras.layers.concatenate([encoded_b1, encoded_d1], axis=-1, name='merged_vector_m') 
     
    merged_vector_lnc = LSTM(32)(merged_vector_lnc)
    merged_vector_m = LSTM(32)(merged_vector_m)
    
    merged_vector = keras.layers.concatenate([merged_vector_lnc, merged_vector_m], axis=-1, name='merged_vector')
    x = Dense(8, kernel_regularizer=regularizers.l2(0.15))(merged_vector)
    dropout = Dropout(0.5)(x)

    sp = Score_pooling(output_dim=1, kernel_regularizer=l2(args.weight_decay), pooling_mode=args.pooling_mode, name='score_pooling')(dropout)
    #sp = Activation('sigmoid')(sp)
    model = Model(inputs=[tweet_a, tweet_b, tweet_c, tweet_d], outputs=[sp])
    #model.summary()
    #sgd = SGD(lr=args.init_lr, decay=1e-4, momentum=args.momentum, nesterov=True)
    #model.compile(loss=bag_loss, optimizer=sgd, metrics=[bag_accuracy])
    nadam=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(loss=bag_loss, optimizer=nadam, metrics=[bag_accuracy])
    return model



def parse_args():
    parser = argparse.ArgumentParser(description='Train a mi-net')
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to train on, like musk1 or fox',
                        default=None, type=str)
    parser.add_argument('--dataset_struct', dest='dataset_struct',
                        help='dataset struct to train on, like musk1 or fox',
                        default=None, type=str)
    parser.add_argument('--pooling', dest='pooling_mode',
                        help='mode of MIL pooling',
                        default='max', type=str)
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=5e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=0.005, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=50, type=int)
    parser.add_argument('--lncRNA_len', dest='lncRNA_len',
                        help='length of lncRNA',
                        default=300, type=int)
    parser.add_argument('--mRNA_len', dest='mRNA_len',
                        help='length of mRNA',
                        default=600, type=int)
    parser.add_argument('--lncRNA_struct_len', dest='lncRNA_struct_len',
                        help='length of lncRNA',
                        default=300, type=int)
    parser.add_argument('--mRNA_struct_len', dest='mRNA_struct_len',
                        help='length of mRNA',
                        default=300, type=int)
    parser.add_argument('--interaction', dest='interaction',
                        help='lncRNA-mRNA pair',
                        default='NULL', type=str)
    parser.add_argument('--pre_trained_weight', dest='pre_trained_weight',
                        help='pre-trained weight file',
                        default='model_deepLPI.h5', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1) 
    
    args = parser.parse_args()
    return args



def test_deepLPI(model, test_set, test_mRNA_set, test_lncRNA_set, test_set_str, test_mRNA_set_str, test_lncRNA_set_str, test_bags_nm, test_ins_nm):
    num_test_batch = len(test_set)
    num_test_mRNA_batch = len(test_mRNA_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    y_p_all=[]
    y_l_all=[]
    idx=0;
    for ibatch, batch in enumerate(test_set):
        if test_set[ibatch][0].shape[0]!=test_set_str[ibatch][0].shape[0]:
            continue
        idx=idx+1
        y_pred_keras = model.predict([test_lncRNA_set[ibatch][0], test_mRNA_set[ibatch][0], test_lncRNA_set_str[ibatch][0], test_mRNA_set_str[ibatch][0]]).ravel()
        y_test=batch[1]
        np.set_printoptions(threshold=sys.maxsize)
        y_p_all.append(np.max(y_pred_keras))
        y_l_all.append(np.max(y_test))
        result = model.test_on_batch([test_lncRNA_set[ibatch][0], test_mRNA_set[ibatch][0] , test_lncRNA_set_str[ibatch][0], test_mRNA_set_str[ibatch][0] ], test_lncRNA_set[ibatch][1])
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]
    auc_keras=roc_auc_score(y_l_all, y_p_all)
    prauc=precision_cal(y_l_all, y_p_all)
    return np.mean(test_loss), np.mean(test_acc), auc_keras, prauc


def unit_test_deepLPI(model, test_set, test_mRNA_set, test_lncRNA_set, test_set_str, test_mRNA_set_str, test_lncRNA_set_str, test_bags_nm, test_ins_nm):
    num_test_batch = len(test_set)
    num_test_mRNA_batch = len(test_mRNA_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    y_p_all=[]
    y_l_all=[]
    idx=0;
    result=0;
    for ibatch, batch in enumerate(test_set):
        if test_set[ibatch][0].shape[0]!=test_set_str[ibatch][0].shape[0]:
            continue
        print(test_bags_nm[ibatch])
        idx=idx+1
        y_pred_keras = model.predict([test_lncRNA_set[ibatch][0], test_mRNA_set[ibatch][0], test_lncRNA_set_str[ibatch][0], test_mRNA_set_str[ibatch][0]]).ravel()
        y_test=batch[1]
        np.set_printoptions(threshold=sys.maxsize)
        y_p_all.append(np.max(y_pred_keras))
        y_l_all.append(np.max(y_test))
        result = model.test_on_batch([test_lncRNA_set[ibatch][0], test_mRNA_set[ibatch][0] , test_lncRNA_set_str[ibatch][0], test_mRNA_set_str[ibatch][0] ], test_lncRNA_set[ibatch][1])
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]
        print(result)
    return result


def run_crf(epoch, score_map, bag_label, bag_index, co_exp_net_isoform, co_exp_net_lncRNA, training_size, testing_size, theta, sigma = 10):
    bag_label = bag_label[0: training_size]
    bag_index = bag_index[0: training_size]
    positive_unary_energy = 1 - score_map

    crf_isoform = CRF(training_size, testing_size, positive_unary_energy, co_exp_net_isoform, theta, bag_label, bag_index)
    crf_lncRNA = CRF(training_size, testing_size, positive_unary_energy, co_exp_net_lncRNA, theta, bag_label, bag_index)
    
    label_update_i, pos_prob_crf_i, unary_potential_i, pairwise_potential_i = crf_isoform.inference(10)
    label_update_l, pos_prob_crf_l, unary_potential_l, pairwise_potential_l = crf_lncRNA.inference(10)
    
    label_update = label_update_i + label_update_l 
    pos_prob_crf = pos_prob_crf_i + pos_prob_crf_l
    unary_potential = unary_potential_i + unary_potential_l
    pairwise_potential = pairwise_potential_i + pairwise_potential_l
    
    if epoch == 0:
        theta_prime_isoform = crf_isoform.parameter_learning(bag_label[0:training_size], theta, sigma)
        theta_prime_lncRNA = crf_lncRNA.parameter_learning(bag_label[0:training_size], theta, sigma)
    else:
        theta_prime_isoform = crf_isoform.parameter_learning(label_update, theta, sigma)
        theta_prime_lncRNA = crf_lncRNA.parameter_learning(label_update, theta, sigma)
    
    theta_prime = theta_prime_isoform + theta_prime_lncRNA
    
    return label_update, theta_prime, pos_prob_crf, unary_potential, pairwise_potential


def extract_data(dataset, dataset_str, lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len):
    train_bags = dataset['train']
    test_bags = dataset['test']

    train_mRNA_bags = dataset['train_mRNA']
    test_mRNA_bags = dataset['test_mRNA']
    train_lncRNA_bags = dataset['train_lncRNA']
    test_lncRNA_bags = dataset['test_lncRNA']

    train_bags_nm = dataset['train_bags_nm']
    train_ins_nm = dataset['train_ins_nm']
    test_bags_nm = dataset['test_bags_nm']
    test_ins_nm = dataset['test_ins_nm']


    train_bags_str = dataset_str['train']
    test_bags_str = dataset_str['test']

    train_mRNA_bags_str = dataset_str['train_mRNA']
    test_mRNA_bags_str = dataset_str['test_mRNA']
    train_lncRNA_bags_str = dataset_str['train_lncRNA']
    test_lncRNA_bags_str = dataset_str['test_lncRNA']

    train_bags_nm_str = dataset_str['train_bags_nm']
    train_ins_nm_str = dataset_str['train_ins_nm']
    test_bags_nm_str = dataset_str['test_bags_nm']
    test_ins_nm_str = dataset_str['test_ins_nm']

    # convert bag to batch
    train_mRNA_set = convertToBatch(train_mRNA_bags)
    test_mRNA_set = convertToBatch(test_mRNA_bags)
    train_lncRNA_set = convertToBatch(train_lncRNA_bags)
    test_lncRNA_set = convertToBatch(test_lncRNA_bags)
    train_set = convertToBatch(train_bags)
    test_set = convertToBatch(test_bags)
    dimension = train_set[0][0].shape[0]

    train_mRNA_set_str = convertToBatch(train_mRNA_bags_str)
    test_mRNA_set_str = convertToBatch(test_mRNA_bags_str)
    train_lncRNA_set_str = convertToBatch(train_lncRNA_bags_str)
    test_lncRNA_set_str = convertToBatch(test_lncRNA_bags_str)
    train_set_str = convertToBatch(train_bags_str)
    test_set_str = convertToBatch(test_bags_str)
    dimension_str = train_set_str[0][0].shape[0]

    return train_bags,test_bags,train_mRNA_bags, test_mRNA_bags, train_lncRNA_bags, test_lncRNA_bags, train_bags_nm, train_ins_nm,test_bags_nm,test_ins_nm, 
    train_bags_str, test_bags_str, train_mRNA_bags_str, test_mRNA_bags_str, train_lncRNA_bags_str, test_lncRNA_bags_str,
    train_bags_nm_str, train_ins_nm_str, test_bags_nm_str, test_ins_nm_str,
    train_mRNA_set, test_mRNA_set, train_lncRNA_set, test_lncRNA_set, train_set, test_set, dimension, 
    train_mRNA_set_str, test_mRNA_set_str, train_lncRNA_set_str, test_lncRNA_set_str, train_set_str, test_set_str, dimension_str


def deepLPI_train(dataset, dataset_str, lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len):
    train_bags = dataset['train']
    test_bags = dataset['test']

    train_mRNA_bags = dataset['train_mRNA']
    test_mRNA_bags = dataset['test_mRNA']
    train_lncRNA_bags = dataset['train_lncRNA']
    test_lncRNA_bags = dataset['test_lncRNA']

    train_bags_nm = dataset['train_bags_nm']
    train_ins_nm = dataset['train_ins_nm']
    test_bags_nm = dataset['test_bags_nm']
    test_ins_nm = dataset['test_ins_nm']


    train_bags_str = dataset_str['train']
    test_bags_str = dataset_str['test']

    train_mRNA_bags_str = dataset_str['train_mRNA']
    test_mRNA_bags_str = dataset_str['test_mRNA']
    train_lncRNA_bags_str = dataset_str['train_lncRNA']
    test_lncRNA_bags_str = dataset_str['test_lncRNA']

    train_bags_nm_str = dataset_str['train_bags_nm']
    train_ins_nm_str = dataset_str['train_ins_nm']
    test_bags_nm_str = dataset_str['test_bags_nm']
    test_ins_nm_str = dataset_str['test_ins_nm']

    # convert bag to batch
    train_mRNA_set = convertToBatch(train_mRNA_bags)
    test_mRNA_set = convertToBatch(test_mRNA_bags)
    train_lncRNA_set = convertToBatch(train_lncRNA_bags)
    test_lncRNA_set = convertToBatch(test_lncRNA_bags)
    train_set = convertToBatch(train_bags)
    test_set = convertToBatch(test_bags)
    dimension = train_set[0][0].shape[0]

    train_mRNA_set_str = convertToBatch(train_mRNA_bags_str)
    test_mRNA_set_str = convertToBatch(test_mRNA_bags_str)
    train_lncRNA_set_str = convertToBatch(train_lncRNA_bags_str)
    test_lncRNA_set_str = convertToBatch(test_lncRNA_bags_str)
    train_set_str = convertToBatch(train_bags_str)
    test_set_str = convertToBatch(test_bags_str)
    dimension_str = train_set_str[0][0].shape[0]

    model = model_func(lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len)

    # train model
    t1 = time.time()
    num_batch = len(train_set)
    all_auc=[]
    all_auprc=[]
    iso_expr_data_all=get_expr_data("./dataset/isoform_expression_data.txt")
    lnc_expr_data_all=get_expr_data("./dataset/lncRNA_expression_data.txt")
    lncRNA_feature_colum=188 #small dataset    
    for epoch in range(args.max_epoch):
        #Training
        initial_score_all = np.array([])
        crf_bag_index=[]
        y_all=np.array([])
        lnc_expr_data=[]
        iso_expr_data=[]
        num_train_batch = len(train_set)
        train_loss = np.zeros((num_train_batch, 1), dtype=float)
        train_acc = np.zeros((num_train_batch, 1), dtype=float)
        for ibatch, batch in enumerate(train_mRNA_set):
            if train_set[ibatch][0].shape[0]!=train_set_str[ibatch][0].shape[0]:
                continue
            y_all=np.hstack((y_all, train_mRNA_set[ibatch][1]))
            initial_score_all_ = model.predict_on_batch([train_lncRNA_set[ibatch][0], train_mRNA_set[ibatch][0], train_lncRNA_set_str[ibatch][0], train_mRNA_set_str[ibatch][0]])
            initial_score_all = np.hstack((initial_score_all, np.transpose(initial_score_all_)[0]))
            i=0
            for i in range(train_mRNA_set[ibatch][0].shape[0]):
                crf_bag_index.append(ibatch)
            ibag_name=train_bags_nm[ibatch].encode('ascii','ignore').strip()
            ibag_name.replace("'","-")
            if len(ibag_name.split('-'))>2:
                lncRNA_name=ibag_name.split('-')[0]+'-'+ibag_name.split('-')[1]
            else:
                lncRNA_name=ibag_name.split('-')[0]
            for ins in train_ins_nm[ibatch]:
                if lncRNA_name in lnc_expr_data_all:
                    lnc_expr_data.append(lnc_expr_data_all[lncRNA_name])
                else:
                    lnc_expr_data.append([0] * lncRNA_feature_colum)
                iso_expr_data.append(iso_expr_data_all[ins.encode('ascii','ignore').strip()])#nicodedata.normalize("NFKD", ins)])

        y_all=np.asarray(y_all, dtype=np.int)

        #WGCNA for isoform expression data
        iso_expr_data=np.asarray(iso_expr_data)
        co_exp_net=np.corrcoef(iso_expr_data)
        # Set nan to be zero
        nan_where = np.isnan(co_exp_net)
        co_exp_net[nan_where] = 0
        # Diagnal to be zero
        for ii in range(co_exp_net.shape[0]):
            co_exp_net[ii, ii] = 0
        # Apply soft threshold
        co_exp_net = np.fabs(co_exp_net)
        co_exp_net = pow(co_exp_net, 6)
        co_exp_net_isoform=co_exp_net

        #WGCNA for lncRNA expression data
        lnc_expr_data=np.asarray(lnc_expr_data)
        lnc_co_exp_net=np.corrcoef(lnc_expr_data)
        # Set nan to be zero
        lnc_nan_where = np.isnan(lnc_co_exp_net)
        lnc_co_exp_net[lnc_nan_where] = 0
        # Diagnal to be zero
        for ii in range(lnc_co_exp_net.shape[0]):
            lnc_co_exp_net[ii, ii] = 0
        # Apply soft threshold
        lnc_co_exp_net = np.fabs(lnc_co_exp_net)
        lnc_co_exp_net = pow(lnc_co_exp_net, 6)
        co_exp_net_lncRNA=lnc_co_exp_net

        crf_bag_index=np.asarray(crf_bag_index)
        K_training_size=y_all.shape[0]
        K_testing_size=0
        theta = np.array([1.0, 1.0])
        new_label, theta, pos_prob_crf, unary_potential, pairwise_potential = run_crf(epoch, initial_score_all, y_all, crf_bag_index, co_exp_net_isoform, co_exp_net_lncRNA, K_training_size, K_testing_size, theta, sigma=0.1)
        if epoch > 0:
            s_index=0
            updated_train_label=[]
            for ibatch, batch in enumerate(train_mRNA_set):
                e_index=s_index+train_lncRNA_set[ibatch][1].shape[0]
                updated_train_label.append((train_lncRNA_set[ibatch][0], np.asarray(new_label[s_index:e_index])))
                s_index=e_index
            train_lncRNA_set=updated_train_label
        for ibatch, batch in enumerate(train_mRNA_set):
            if train_set[ibatch][0].shape[0]!=train_set_str[ibatch][0].shape[0] : continue
            if train_set[ibatch][0].shape[0]!=train_lncRNA_set[ibatch][1].shape[0]: continue
            result = model.train_on_batch([train_lncRNA_set[ibatch][0], train_mRNA_set[ibatch][0], train_lncRNA_set_str[ibatch][0], train_mRNA_set_str[ibatch][0]], train_lncRNA_set[ibatch][1])
            train_loss[ibatch] = result[0]
            train_acc[ibatch] = result[1]
            model, mean_train_loss, mean_train_acc = model, np.mean(train_loss), np.mean(train_acc)
    
    return model

def deepLPI(dataset, dataset_str, lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len, pre_trained_weight):
    train_bags = dataset['train']
    test_bags = dataset['test']
    
    train_mRNA_bags = dataset['train_mRNA']
    test_mRNA_bags = dataset['test_mRNA']
    train_lncRNA_bags = dataset['train_lncRNA']
    test_lncRNA_bags = dataset['test_lncRNA']

    train_bags_nm = dataset['train_bags_nm']
    train_ins_nm = dataset['train_ins_nm']
    test_bags_nm = dataset['test_bags_nm'] 
    test_ins_nm = dataset['test_ins_nm']


    train_bags_str = dataset_str['train']
    test_bags_str = dataset_str['test']

    train_mRNA_bags_str = dataset_str['train_mRNA']
    test_mRNA_bags_str = dataset_str['test_mRNA']
    train_lncRNA_bags_str = dataset_str['train_lncRNA']
    test_lncRNA_bags_str = dataset_str['test_lncRNA']

    train_bags_nm_str = dataset_str['train_bags_nm']
    train_ins_nm_str = dataset_str['train_ins_nm']
    test_bags_nm_str = dataset_str['test_bags_nm']
    test_ins_nm_str = dataset_str['test_ins_nm']

    # convert bag to batch
    train_mRNA_set = convertToBatch(train_mRNA_bags)
    test_mRNA_set = convertToBatch(test_mRNA_bags)
    train_lncRNA_set = convertToBatch(train_lncRNA_bags)
    test_lncRNA_set = convertToBatch(test_lncRNA_bags)
    train_set = convertToBatch(train_bags)
    test_set = convertToBatch(test_bags)    
    dimension = train_set[0][0].shape[0] 
    
    train_mRNA_set_str = convertToBatch(train_mRNA_bags_str)
    test_mRNA_set_str = convertToBatch(test_mRNA_bags_str)
    train_lncRNA_set_str = convertToBatch(train_lncRNA_bags_str)
    test_lncRNA_set_str = convertToBatch(test_lncRNA_bags_str)
    train_set_str = convertToBatch(train_bags_str)
    test_set_str = convertToBatch(test_bags_str)    
    dimension_str = train_set_str[0][0].shape[0]

    
    model = model_func(lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len)
    
    ## Load model
    #model.load_weights('model_deepLPI.h5')
    #model.save('testsave.h5')
    #visulize_saliency(model, test_lncRNA_set[0][0], test_mRNA_set[0][0], test_lncRNA_set[0][1]) 

    t1 = time.time()
    num_batch = len(train_set)
    all_auc=[]
    all_auprc=[]
    
    model=deepLPI_train(dataset, dataset_str, lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len) 
    #model.load_weights('model_deepLPI.h5')
    model.load_weights(pre_trained_weight)
    for epoch in range(1):
        test_loss, test_acc, test_auc, test_auprc = test_deepLPI(model, test_set, test_mRNA_set, test_lncRNA_set, test_set_str, test_mRNA_set_str, test_lncRNA_set_str, test_bags_nm, test_ins_nm)
        all_auc.append(test_auc)
        all_auprc.append(test_auprc)
    
    
    t2 = time.time()
    
    #model.save('model_deepLPI.h5')
    return test_acc, np.mean(all_auc), np.mean(all_auprc)

bagname=[]


def deepLPI_unit(dataset, dataset_str, bagdict, lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len):
    train_bags = dataset['train']
    test_bags = dataset['test']

    train_mRNA_bags = dataset['train_mRNA']
    test_mRNA_bags = dataset['test_mRNA']
    train_lncRNA_bags = dataset['train_lncRNA']
    test_lncRNA_bags = dataset['test_lncRNA']

    train_bags_nm = dataset['train_bags_nm']
    train_ins_nm = dataset['train_ins_nm']
    test_bags_nm = dataset['test_bags_nm']
    test_ins_nm = dataset['test_ins_nm']


    train_bags_str = dataset_str['train']
    test_bags_str = dataset_str['test']

    train_mRNA_bags_str = dataset_str['train_mRNA']
    test_mRNA_bags_str = dataset_str['test_mRNA']
    train_lncRNA_bags_str = dataset_str['train_lncRNA']
    test_lncRNA_bags_str = dataset_str['test_lncRNA']

    train_bags_nm_str = dataset_str['train_bags_nm']
    train_ins_nm_str = dataset_str['train_ins_nm']
    test_bags_nm_str = dataset_str['test_bags_nm']
    test_ins_nm_str = dataset_str['test_ins_nm']

    # convert bag to batch
    train_mRNA_set = convertToBatch(train_mRNA_bags)
    test_mRNA_set = convertToBatch(test_mRNA_bags)
    train_lncRNA_set = convertToBatch(train_lncRNA_bags)
    test_lncRNA_set = convertToBatch(test_lncRNA_bags)
    train_set = convertToBatch(train_bags)
    test_set = convertToBatch(test_bags)
    dimension = train_set[0][0].shape[0]

    train_mRNA_set_str = convertToBatch(train_mRNA_bags_str)
    test_mRNA_set_str = convertToBatch(test_mRNA_bags_str)
    train_lncRNA_set_str = convertToBatch(train_lncRNA_bags_str)
    test_lncRNA_set_str = convertToBatch(test_lncRNA_bags_str)
    train_set_str = convertToBatch(train_bags_str)
    test_set_str = convertToBatch(test_bags_str)
    dimension_str = train_set_str[0][0].shape[0]
    

    
    idx=0
    for ibatch, batch in enumerate(test_set):
        if test_set[ibatch][0].shape[0]!=test_set_str[ibatch][0].shape[0]:
            continue
        idx=idx+1
        bagdict[test_bags_nm[ibatch].encode('ascii','ignore').strip()]=([test_lncRNA_set[ibatch][0], test_mRNA_set[ibatch][0], test_lncRNA_set_str[ibatch][0], test_mRNA_set_str[ibatch][0]], test_lncRNA_set[ibatch][1])
        bagname.append(test_bags_nm[ibatch].encode('ascii','ignore').strip())
    idx=0
    for ibatch, batch in enumerate(train_set):
        if train_set[ibatch][0].shape[0]!=train_set_str[ibatch][0].shape[0]:
            continue
        idx=idx+1
        bagdict[train_bags_nm[ibatch].encode('ascii','ignore').strip()]=([train_lncRNA_set[ibatch][0], train_mRNA_set[ibatch][0], train_lncRNA_set_str[ibatch][0], train_mRNA_set_str[ibatch][0]], test_lncRNA_set[ibatch][1])
        bagname.append(train_bags_nm[ibatch].encode('ascii','ignore').strip())

    return bagdict


def loadmodel(lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len, pre_trained_weight):
    model = model_func(lncRNA_len, mRNA_len, lncRNA_struct_len, mRNA_struct_len)
    #model.load_weights('model_deepLPI.h5')
    model.load_weights(pre_trained_weight)
    return model

def interactiontest(args):
    dataset = load_dataset(args.dataset, 2, args.lncRNA_len, args.mRNA_len)
    dataset_struct = load_dataset(args.dataset_struct, 2, args.lncRNA_len, args.mRNA_len)
    
    bagdict={}
    for ifold in range(2):
        bagdict = deepLPI_unit(dataset[ifold], dataset_struct[ifold], bagdict, args.lncRNA_len, args.mRNA_len,  args.lncRNA_struct_len, args.mRNA_struct_len)

    model=loadmodel(args.lncRNA_len, args.mRNA_len,  args.lncRNA_struct_len, args.mRNA_struct_len, args.pre_trained_weight)
    #name="TUG1-PET117"
    interaction="MALAT1-TMEM69"
    interaction="TUG1-EHD2"
    interaction='MALAT1-ANKRD1' #'Interacted pair')
    interaction='TUG1-BTG3'     #'Interacted pair')
    interaction=args.interaction
    try:
        result = model.test_on_batch(bagdict[interaction][0], bagdict[interaction][1])
    except:
        #print(interaction, " : Non-interacted pair")
        return
    unit_pred=result[0]
    if(unit_pred>2): print(interaction, " : Interacted pair")
    else: print(interaction, " : Non-interacted pair")
    return

def train(args):
    #perform cross-validation experiments
    run = 1
    n_folds = 5#10
    acc = np.zeros((run, n_folds), dtype=float)
    auc = np.zeros((run, n_folds), dtype=float)
    prauc = np.zeros((run, n_folds), dtype=float)
    maxauc=0
    for irun in range(run):
        dataset = load_dataset(args.dataset, n_folds, args.lncRNA_len, args.mRNA_len)
        dataset_struct = load_dataset(args.dataset_struct, n_folds, args.lncRNA_len, args.mRNA_len)
        for ifold in range(n_folds):
            print 'run=', irun, '  fold=', ifold
            acc[irun][ifold], auc[irun][ifold], prauc[irun][ifold] = deepLPI(dataset[ifold], dataset_struct[ifold],  args.lncRNA_len, args.mRNA_len,  args.lncRNA_struct_len, args.mRNA_struct_len, args.pre_trained_weight)
    print 'auc = ', np.mean(auc)
    print 'prauc= ', np.mean(prauc)

if __name__ == '__main__':
    #def main_prog():
    args = parse_args()
    
    #unit_test
    n_folds=5
    dataset = load_dataset(args.dataset, n_folds, args.lncRNA_len, args.mRNA_len)
    dataset_struct = load_dataset(args.dataset_struct, n_folds, args.lncRNA_len, args.mRNA_len)
    
    if(args.interaction=='NULL'):
        train(args)
    else: 
        interactiontest(args)
    

