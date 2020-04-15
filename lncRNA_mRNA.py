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

import sys
import time
from random import shuffle
import numpy as np
import argparse

from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Input, Dense, Layer, Dropout

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
from itertools import permutations
import itertools
from sklearn.preprocessing import normalize



def encode_n_mer(s, n, alphabets):
    #perms = [''.join(p) for p in permutations('ACGT')]
    #alphabets = ['A', 'C', 'G', 'T']
    perms = [''.join(i) for i in itertools.product(alphabets, repeat = n)]
    perm_dict={}
    for i, p in enumerate(perms):
        perm_dict[p]=i
    s_encode=[]
    i=0
    while (i+n)<len(s):
        flag=0;
        for c in s[i:i+n]:
            if c not in alphabets:
                flag=1;
        if(flag==0):
            s_encode.append(perm_dict[s[i:i+n]])
        i=i+n
    
    #s_encode=np.asarray(s_encode)
    np.set_printoptions(threshold=np.inf)
    return s_encode

def read_lncRNAfile(filename):
    flag=0
    lncRNA_dict={}
    
    with open (filename, "r") as f:
	lncRNA_name = ''
	s = ''
	for line in f:
	    line = line.rstrip()
	    if line[0]=='>': #or line.startswith('>')
                if(flag==1):
                    s=encode_n_mer(s,4,['A', 'C', 'G', 'T'])
                    lncRNA_dict[lncRNA_name]=s;
		lncRNA_name=line.split('|')[5]
		s=''
	    else:
		s = s + line
		flag=1;
    #w = csv.writer(open("lncRNA_seqs4.txt", "w"))
    #for key, val in lncRNA_dict.items():
    #	w.writerow([key, val])
    
    return lncRNA_dict


def read_mRNAfile(filename):
    flag=0
    mRNA_dict={}
    mRNA_name=''
    isoform_name=''
    mRNA_isoform_name_dict={}
    ensmble_isoform_name=''
    refseq_isoform_name=''
    mRNA_ensmble_isoform_name_dict={}
    mRNA_refseq_isoform_name_dict={}
    
    ensmble_refseq_id_dict={}
    ensmble_refseq_id_reader = csv.reader(open("ensmble_refseq_ids.txt", "r"))
    for ensmbleid, refseqid in ensmble_refseq_id_reader:
        ensmble_refseq_id_dict[ensmbleid]=refseqid

    expression_avail_refseqid_name_list=[]
    expression_avail_refseqid_name_dict={}
    expression_avail_refseqid_name_reader=csv.reader(open("refseqid_isoform_expression.txt", "r"))
    for refseq in expression_avail_refseqid_name_reader:
        expression_avail_refseqid_name_list.append(refseq[0].split('.')[0])

    with open (filename, "r") as f:
	mRNA_name = ''
	s = ''
	for line in f:
	    line = line.rstrip()
	    if(len(line)==0):
		continue;
	    if line[0]=='>': #or line.startswith('>')
		if(flag==1 and ensmble_refseq_id_dict[ensmble_isoform_name]):  #isoform_name ):
                    if ensmble_refseq_id_dict[ensmble_isoform_name] in expression_avail_refseqid_name_list:
		        if mRNA_name in mRNA_dict:
                            s=encode_n_mer(s,3,['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V'])
			    mRNA_dict[mRNA_name].append(s);
                            mRNA_isoform_name_dict[mRNA_name].append(isoform_name)
                            mRNA_ensmble_isoform_name_dict[mRNA_name].append(ensmble_isoform_name)
		    	    mRNA_refseq_isoform_name_dict[mRNA_name].append(refseq_isoform_name)
		        else:
                            s=encode_n_mer(s,3,['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V'])
		    	    mRNA_dict[mRNA_name]=[s];
                            mRNA_isoform_name_dict[mRNA_name]=[isoform_name]
                            mRNA_ensmble_isoform_name_dict[mRNA_name]=[ensmble_isoform_name]
		            mRNA_refseq_isoform_name_dict[mRNA_name]=[refseq_isoform_name]
		mRNA_name=line.split('|')[5]
                isoform_name=line.split('|')[4]
                ensmble_isoform_name=line.split('|')[0]
		ensmble_isoform_name=ensmble_isoform_name.split('>')[1]
                refseq_isoform_name=ensmble_refseq_id_dict[ensmble_isoform_name]
		s=''
	    else:
		s = s + line
		flag=1;
    
    #w = csv.writer(open("protein_seqs.txt", "w"))
    #for key, val in mRNA_dict.items():
    #	w.writerow([key, val])
    
    #w = csv.writer(open("gene_isoform.txt", "w"))
    #for key, val in mRNA_isoform_name_dict.items():
    #    w.writerow([key, val])

    #w = csv.writer(open("gene_ensmble_isoform.txt", "w"))
    #for key, val in mRNA_ensmble_isoform_name_dict.items():
    #    w.writerow([key, val])

    return mRNA_dict, mRNA_refseq_isoform_name_dict



def get_lncRNA_mRNA_pair(filename, lncRNA_dict, mRNA_dict, mRNA_refseq_isoform_name_dict):
    
    int=pd.read_table(filename)
    int2=int
    
    seqs_mRNA = []
    seqs_lncRNA = []
    lncRNA_name_list = []
    gene_name_list = []
    gene_lncRNA_name_list = []
    isoform_name_list = []
    
    for index, row in int2.iterrows():
	if (row['lncRNA'] in lncRNA_dict and row['gene'] in mRNA_dict):
	    i=0
	    for mRNA in mRNA_dict[row['gene']]:
		lncRNA_name_list.append(row['lncRNA'])
		gene_name_list.append(row['gene'])
		seqs_lncRNA.append(lncRNA_dict[row['lncRNA']])
		seqs_mRNA.append(mRNA)
		gene_lncRNA_name_list.append(str(row['lncRNA'])+str("-")+str(row['gene']))
		i=i+1
            for isoform_name in mRNA_refseq_isoform_name_dict[row['gene']]:
                isoform_name_list.append(str(isoform_name))
    return seqs_mRNA, seqs_lncRNA, gene_name_list, lncRNA_name_list, gene_lncRNA_name_list, isoform_name_list
    


def rna_encoding(seqs_lncRNA):
    #print "RNA encoding"
    CHARS = 'ACGT'
    CHARS_COUNT = len(CHARS)

    maxlen = 300#max(map(len, seqs))
    res = np.zeros((len(seqs_lncRNA), maxlen), dtype=np.uint8)

    for si, seq in enumerate(seqs_lncRNA):
        seqlen = len(seq)
	for i, schar in enumerate(seq):
            if i<maxlen:
                res[si][i] = schar#ord(schar)
    np.set_printoptions(threshold=np.inf)
    return res



def protein_encoding(seqs_mRNA):
    #print "Protein encoding"
    CHARS = 'ARNDCEQGHILKMFPSTWYV'
    CHARS_COUNT = len(CHARS)

    maxlen = 600#max(map(len, seqs))
    res_gene = np.zeros((len(seqs_mRNA), maxlen), dtype=np.uint8)

    for si, seq in enumerate(seqs_mRNA):
	seqlen = len(seq)
	for i, schar in enumerate(seq):
	    if i<maxlen:
		res_gene[si][i] = schar#ord(schar)

    np.set_printoptions(threshold=np.inf)
    return res_gene



def model_func(data_a, data_b, labels):
    tweet_a = Input(shape=(200,)) #50, 4))
    encoded_a1=Embedding(input_dim=4, output_dim=16)(tweet_a)
    encoded_a1=LSTM(8)(encoded_a1)
    encoded_a1=Dense(4, kernel_regularizer=regularizers.l2(0.15))(encoded_a1)
    encoded_a1=Activation('relu')(encoded_a1)
    encoded_a1=Dropout(0.5)(encoded_a1)
    encoded_a1=Dense(4, kernel_regularizer=regularizers.l2(0.15))(encoded_a1)
    encoded_a =Activation('relu')(encoded_a1)

    tweet_b = Input(shape=(1000,))
    encoded_b1=Embedding(input_dim=21, output_dim=8)(tweet_b)
    encoded_b1 = Convolution1D(filters = 64, kernel_size = 16, strides = 1,  padding = 'valid', activation = 'relu')(encoded_b1)
    encoded_b1=Flatten()(encoded_b1)
    encoded_b1=Dense(32, kernel_regularizer=regularizers.l2(0.15))(encoded_b1)
    encoded_b1=Activation('relu')(encoded_b1)
    encoded_b1=Dropout(0.5)(encoded_b1)
    encoded_b1=Dense(16, kernel_regularizer=regularizers.l2(0.15))(encoded_b1)
    encoded_b =Activation('relu')(encoded_b1)

    merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1, name='merged_vector')

    x = Dense(16, kernel_regularizer=regularizers.l2(0.15))(merged_vector)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, kernel_regularizer=regularizers.l2(0.15))(x)
    predictions = Activation('sigmoid')(x)

    model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)
    model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit([data_a, data_b], labels, epochs=3)


    layer_name = 'merged_vector'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict([data_a, data_b])

    return intermediate_output
















lncRNA_dict={}
mRNA_dict={}

lncRNA_dict=read_lncRNAfile("gencode.v28.lncRNA_transcripts.fa")
mRNA_dict, mRNA_isoform_name_dict = read_mRNAfile("Protein_Sequence_of_Ensemble_Gene.txt")

seqs_mRNA = []
seqs_lncRNA = []    
lncRNA_name_list = []
gene_name_list = []        
gene_lncRNA_name_list = []
isoform_name_list = []

#seqs_mRNA, seqs_lncRNA, gene_name_list, lncRNA_name_list, gene_lncRNA_name_list, isoform_name_list = get_lncRNA_mRNA_pair('all_interation_mRna_lncRNA.txt', lncRNA_dict, mRNA_dict, mRNA_isoform_name_dict)
seqs_mRNA, seqs_lncRNA, gene_name_list, lncRNA_name_list, gene_lncRNA_name_list, isoform_name_list = get_lncRNA_mRNA_pair('interaction_count.txt', lncRNA_dict, mRNA_dict, mRNA_isoform_name_dict)

data_a=rna_encoding(seqs_lncRNA)
data_b=protein_encoding(seqs_mRNA)
labels = np.asarray([[np.random.randint(1,2)] for p in range(0,len(seqs_lncRNA))])

#intermediate_output = model_func(data_a, data_b, labels)


lncRNA_dict_ni=lncRNA_dict
mRNA_dict_ni=mRNA_dict

seqs_lncRNA_ni=[]
seqs_mRNA_ni=[]
lncRNA_name_list_ni = []
gene_name_list_ni = []   
gene_lncRNA_name_list_ni=[]
isoform_name_list_ni=[]

#seqs_mRNA_ni, seqs_lncRNA_ni, gene_name_list_ni, lncRNA_name_list_ni, gene_lncRNA_name_list_ni, isoform_name_list_ni = get_lncRNA_mRNA_pair('all_non_interaction_mRNA_lncRNA_within.txt', lncRNA_dict, mRNA_dict, mRNA_isoform_name_dict)
seqs_mRNA_ni, seqs_lncRNA_ni, gene_name_list_ni, lncRNA_name_list_ni, gene_lncRNA_name_list_ni, isoform_name_list_ni = get_lncRNA_mRNA_pair('non_interaction_count.txt', lncRNA_dict, mRNA_dict, mRNA_isoform_name_dict)

data_a_ni=rna_encoding(seqs_lncRNA_ni)
data_b_ni=protein_encoding(seqs_mRNA_ni)
labels_ni = np.asarray([[np.random.randint(0,1)] for p in range(0,len(seqs_lncRNA_ni))])


data_i=np.hstack((data_a, data_b))
data_ni=np.hstack((data_a_ni, data_b_ni))
io=np.vstack((data_i, data_ni))

lab=np.vstack((labels, labels_ni))
gene=gene_name_list+ gene_name_list_ni  
lnc=lncRNA_name_list+ lncRNA_name_list_ni
gene_lnc=gene_lncRNA_name_list+gene_lncRNA_name_list_ni
isoform=isoform_name_list+isoform_name_list_ni



sio.savemat('./dataset/merged_lncRNA_protein_mini.mat', {'x':{'data':io, 'nlab':lab,'ident':{'ident':isoform, 'milbag':gene_lnc}}})



