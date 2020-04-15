import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import numpy as np
from keras.utils import to_categorical
from numpy import array

import numpy as np
import pandas as pd
import csv
import sys
csv.field_size_limit(sys.maxsize)
import scipy.io as sio
import ast


def encode(sequence, struct_prob, mlen):
    CHARS = 'SHIFTM'
    CHARS_COUNT = len(CHARS)
    maxlen = mlen
    res = np.zeros((maxlen, CHARS_COUNT), dtype=np.float32)#np.uint8)

    for si, seq in enumerate(sequence):
        seqlen = len(seq)
        arr = np.chararray((seqlen,), buffer=seq)
        arr4 = np.chararray((CHARS_COUNT,), buffer=CHARS)
        for i, schar in enumerate(seq):
            for ii, char in enumerate(CHARS):
                if schar == char:
                    if i<maxlen:
                        res[i][ii] = res[i][ii] + float(struct_prob[si])
    return res





int=pd.read_table('interaction_count.txt')
int2=int


lncRNA_struct_dict={}
mRNA_struct_dict={}

lncRNA_struct_reader = csv.reader(open("lncRNA_eden_struct.txt", "r"))
for key, lncRNAshapes in lncRNA_struct_reader:
    lncRNA = ast.literal_eval(lncRNAshapes)
    lncRNA_struct_dict[key]=lncRNA
 
mRNA_struct_reader = csv.reader(open("mRNA_eden_struct.txt", "r"))
for key, mRNAshapes in mRNA_struct_reader:
    mRNA = ast.literal_eval(mRNAshapes)
    mRNA_struct_dict[key]=mRNA



structs_mRNA = []
structs_lncRNA = []    
lncRNA_name_list = []
gene_name_list = []
gene_lncRNA_name_list = []
isoform_name_list = []

for index, row in int2.iterrows():
    if (row['lncRNA'] in lncRNA_struct_dict and row['gene'] in mRNA_struct_dict):
        isoform_count=row['#isoform']
        i=0
        for mRNA in mRNA_struct_dict[row['gene']]:
            lncRNA_name_list.append(row['lncRNA'])
            gene_name_list.append(row['gene'])
            gene_lncRNA_name_list.append(str(row['lncRNA'])+str("-")+str(row['gene']))
            isoform_name_list.append(str(row['gene'])+str("-")+str(i))
            i=i+1
            structs_lncRNA.append(lncRNA_struct_dict[row['lncRNA']])
            structs_mRNA.append(mRNA)



s_CHARS = 'SHIFTM'
s_CHARS_COUNT = len(s_CHARS)
maxlen = 600
lnclen=300
mlen=300
struct_data_a = np.zeros((len(structs_lncRNA), lnclen, s_CHARS_COUNT), dtype=np.float32)

for si, struct_prob in enumerate(structs_lncRNA):
    struct_data_a[si]=encode(struct_prob[0][0], struct_prob[0][1], lnclen) 



s_CHARS = 'SHIFTM'
s_CHARS_COUNT = len(s_CHARS)
struct_data_b = np.zeros((len(structs_mRNA), mlen, s_CHARS_COUNT), dtype=np.float32)

for si, struct_prob in enumerate(structs_mRNA):
    struct_data_b[si]=encode(struct_prob[0], struct_prob[1], mlen)


data_a=struct_data_a*0.1
data_b=struct_data_b


labels = np.asarray([[np.random.randint(1,2)] for p in range(0,len(structs_lncRNA))])
tweet_a = Input(shape=(lnclen, 6))
tweet_b = Input(shape=(mlen, 6))


shared_lstm = LSTM(64)


encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1, name='merged_vector')

predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit([data_a, data_b], labels, epochs=2)


layer_name = 'merged_vector'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict([data_a, data_b])


merged_name=np.asarray(zip(lncRNA_name_list, gene_name_list))
merged_data_label=np.hstack((intermediate_output, labels))

merged_dataset=np.hstack((merged_name, merged_data_label))






#sys.exit()











int_ni=pd.read_table('non_interaction_count.txt')
int2_ni=int_ni

lncRNA_struct_dict_ni={}
mRNA_struct_dict_ni={}

lncRNA_struct_reader = csv.reader(open("lncRNA_eden_struct.txt", "r"))
for key, lncRNAshapes in lncRNA_struct_reader:
    lncRNA = ast.literal_eval(lncRNAshapes)
    lncRNA_struct_dict_ni[key]=lncRNA

mRNA_struct_reader = csv.reader(open("mRNA_eden_struct.txt", "r"))
for key, mRNAshapes in mRNA_struct_reader:
    mRNA = ast.literal_eval(mRNAshapes)
    mRNA_struct_dict_ni[key]=mRNA



structs_mRNA_ni = []
structs_lncRNA_ni = []
lncRNA_name_list_ni = []
gene_name_list_ni = []
gene_lncRNA_name_list_ni=[]
isoform_name_list_ni=[]

for index, row in int2_ni.iterrows():
    if (row['lncRNA'] in lncRNA_struct_dict_ni and row['gene'] in mRNA_struct_dict_ni):
        isoform_count = row['#isoform']
        i=0
        for mRNA in mRNA_struct_dict_ni[row['gene']]:
            lncRNA_name_list_ni.append(row['lncRNA'])
            gene_name_list_ni.append(row['gene'])
            gene_lncRNA_name_list_ni.append(str(row['lncRNA'])+str("-")+str(row['gene']))
            isoform_name_list_ni.append(str(row['gene'])+str("-")+str(i))
            i=i+1
            structs_lncRNA_ni.append(lncRNA_struct_dict_ni[row['lncRNA']])
            structs_mRNA_ni.append(mRNA)

s_CHARS = 'SHIFTM'
s_CHARS_COUNT = len(s_CHARS)
maxlen = 600
struct_data_a_ni = np.zeros((len(structs_lncRNA_ni), lnclen, s_CHARS_COUNT), dtype=np.float32)

for si, struct_prob in enumerate(structs_lncRNA_ni):
    struct_data_a_ni[si]=encode(struct_prob[0][0], struct_prob[0][1], lnclen)



s_CHARS = 'SHIFTM'
s_CHARS_COUNT = len(s_CHARS)
maxlen = 600
struct_data_b_ni = np.zeros((len(structs_mRNA_ni), mlen, s_CHARS_COUNT), dtype=np.float32)

for si, struct_prob in enumerate(structs_mRNA_ni):
    struct_data_b_ni[si]=encode(struct_prob[0], struct_prob[1], mlen)


data_a_ni=struct_data_a_ni*0.1
data_b_ni=struct_data_b_ni




labels_ni = np.asarray([[np.random.randint(0,1)] for p in range(0,len(structs_lncRNA_ni))])

tweet_a_ni = Input(shape=(lnclen, 6))
tweet_b_ni = Input(shape=(mlen, 6))






shared_lstm = LSTM(64)


encoded_a_ni = shared_lstm(tweet_a_ni)
encoded_b_ni = shared_lstm(tweet_b_ni)

merged_vector_ni = keras.layers.concatenate([encoded_a_ni, encoded_b_ni], axis=-1, name='merged_vector')

predictions_ni = Dense(1, activation='sigmoid')(merged_vector_ni)

model_ni = Model(inputs=[tweet_a_ni, tweet_b_ni], outputs=predictions_ni)

model_ni.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model_ni.fit([data_a_ni, data_b_ni], labels_ni, epochs=2)


layer_name_ni = 'merged_vector_ni'
intermediate_layer_model_ni = Model(inputs=model_ni.input, outputs=model_ni.get_layer(layer_name).output)
intermediate_output_ni = intermediate_layer_model_ni.predict([data_a_ni, data_b_ni])

#merged_data_label_ni=np.hstack((intermediate_output_ni, labels_ni))
#merged_dataset_ni=np.hstack((merged_name_ni, merged_data_label_ni))


import scipy.io as sio


io_lstm=np.vstack((intermediate_output, intermediate_output_ni))

data_i=np.hstack((data_a, data_b))
data_ni=np.hstack((data_a_ni, data_b_ni))
io=np.vstack((data_i, data_ni))
lab=np.vstack((labels, labels_ni))
gene=gene_name_list+ gene_name_list_ni  
lnc=lncRNA_name_list+ lncRNA_name_list_ni
gene_lnc=gene_lncRNA_name_list+gene_lncRNA_name_list_ni
isoform=isoform_name_list+isoform_name_list_ni

sio.savemat('./dataset/merged_struct_lncRNA_protein_mini3.mat', {'x':{'data':io, 'nlab':lab,'ident':{'ident':isoform, 'milbag':gene_lnc}}})


