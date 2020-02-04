import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import numpy as np
from keras.utils import to_categorical
from numpy import array

import numpy as np
import pandas as pd
import csv
import scipy.io as sio

 
int2=pd.read_table('interation_mRna_lncRNA.txt')
lnc_RNA = []
c=0
flag=0
lncRNA_dict={}


with open ("gencode.v28.lncRNA_transcripts.fa", "r") as f:
    lncRNA_name = ''
    s = ''
    for line in f:
        line = line.rstrip()
        print(line)
        if line[0]=='>': #or line.startswith('>')
            if(flag==1):
                lncRNA_dict[lncRNA_name]=s;
            lncRNA_name=line.split('|')[5]
            s=''
        else:
            s = s + line
            flag=1;


w = csv.writer(open("lncRNA_seqs.txt", "w"))
for key, val in lncRNA_dict.items():
    w.writerow([key, val])


mRNA_exon = []
flag=0
mRNA_dict={}
mRNA_name=''
with open ("Transcript_Sequence_of_RefSeq_filtered_Ensemble_Gene.txt", "r") as f:
    mRNA_name = ''
    s = ''
    for line in f:
        line = line.rstrip()
        if(len(line)==0):
            continue;
        if line[0]=='>': #or line.startswith('>')
            if(flag==1):
                if mRNA_name in mRNA_dict: 
                    mRNA_dict[mRNA_name].append(s);
                else:
                    mRNA_dict[mRNA_name]=[s];
                print(mRNA_name, s)
            mRNA_name=line.split('|')[5]
            s=''
        else:
            s = s + line
            flag=1;

w = csv.writer(open("mRNA_seqs.txt", "w"))
for key, val in mRNA_dict.items():
    w.writerow([key, val])
        
        

seqs_mRNA = []
seqs_lncRNA = []    
lncRNA_name_list = []
gene_name_list = []        


for index, row in int2.iterrows():
    if (row['lncRNA'] in lncRNA_dict and row['gene'] in mRNA_dict):
        for mRNA in mRNA_dict[row['gene']]:
            lncRNA_name_list.append(row['lncRNA'])
            gene_name_list.append(row['gene'])
            seqs_lncRNA.append(lncRNA_dict[row['lncRNA']])
            seqs_mRNA.append(mRNA)

CHARS = 'ACGT'
CHARS_COUNT = len(CHARS)
            
maxlen = 1500#max(map(len, seqs))
res = np.zeros((len(seqs_lncRNA), maxlen, CHARS_COUNT), dtype=np.uint8)
            
for si, seq in enumerate(seqs_lncRNA):
    seqlen = len(seq)
    arr = np.chararray((seqlen,), buffer=seq)
    arr4 = np.chararray((CHARS_COUNT,), buffer=CHARS)
    for i, schar in enumerate(seq):
        for ii, char in enumerate(CHARS):
            if schar == char:
                if i<1500:
                    res[si][i][ii] = 1
                    

data_a=res

CHARS = 'ACGT'
CHARS_COUNT = len(CHARS)
            
maxlen = 1500#max(map(len, seqs))
res_gene = np.zeros((len(seqs_mRNA), maxlen, CHARS_COUNT), dtype=np.uint8)
            
for si, seq in enumerate(seqs_mRNA):
    seqlen = len(seq)
    arr = np.chararray((seqlen,), buffer=seq)
    arr4 = np.chararray((CHARS_COUNT,), buffer=CHARS)
    for i, schar in enumerate(seq):
        for ii, char in enumerate(CHARS):
            if schar == char:
                if i<1500:
                    res_gene[si][i][ii] = 1
                    


print res_gene                   
data_b=res_gene#[0:100]


labels = np.asarray([[np.random.randint(1,2)] for p in range(0,len(seqs_lncRNA))])




tweet_a = Input(shape=(1500, 4))
tweet_b = Input(shape=(1500, 4))


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

print(intermediate_output)

merged_name=np.asarray(zip(lncRNA_name_list, gene_name_list))
merged_data_label=np.hstack((intermediate_output, labels))

merged_dataset=np.hstack((merged_name, merged_data_label))


















int2_ni=pd.read_table('non_interaction_mRNA_lncRNA.txt')

lncRNA_dict_ni={}
mRNA_dict_ni={}
flag=0;
lncRNA_name = ''

with open ("gencode.v28.lncRNA_transcripts.fa", "r") as f:
    lncRNA_name = ''
    s = ''
    for line in f:
        line = line.rstrip()
        print(line)
        if line[0]=='>': #or line.startswith('>')
            if(flag==1):
                lncRNA_dict_ni[lncRNA_name]=s;
            lncRNA_name=line.split('|')[5]
            s=''
        else:
            s = s + line
            flag=1;

w = csv.writer(open("lncRNA_seqs_ni.txt", "w"))
for key, val in lncRNA_dict_ni.items():
    w.writerow([key, val])







flag=0
mRNA_name=''

with open ("Transcript_Sequence_of_RefSeq_filtered_Ensemble_Gene.txt", "r") as f:
    mRNA_name = ''
    s = ''
    for line in f:
        line = line.rstrip()
        if(len(line)==0):
            continue;
        if line[0]=='>': #or line.startswith('>')
            if(flag==1):
                if mRNA_name in mRNA_dict_ni: 
                    mRNA_dict_ni[mRNA_name].append(s);
                else:
                    mRNA_dict_ni[mRNA_name]=[s];
                print(mRNA_name, s)
            mRNA_name=line.split('|')[5]
            s=''
        else:
            s = s + line
            flag=1;

w = csv.writer(open("mRNA_seqs_ni.txt", "w"))
for key, val in mRNA_dict_ni.items():
    w.writerow([key, val])
        

seqs_lncRNA_ni=[]
seqs_mRNA_ni=[]
lncRNA_name_list_ni = []
gene_name_list_ni = []   

for index, row in int2_ni.iterrows():
    if (row['lncRNA'] in lncRNA_dict_ni and row['gene'] in mRNA_dict_ni):
        print row['lncRNA'], row['gene']
        for mRNA in mRNA_dict_ni[row['gene']]:
            lncRNA_name_list_ni.append(row['lncRNA'])
            gene_name_list_ni.append(row['gene'])
            seqs_lncRNA_ni.append(lncRNA_dict_ni[row['lncRNA']])
            seqs_mRNA_ni.append(mRNA)


CHARS = 'ACGT'
CHARS_COUNT = len(CHARS)

maxlen = 1500#max(map(len, seqs))
res_ni = np.zeros((len(seqs_lncRNA_ni), maxlen, CHARS_COUNT), dtype=np.uint8)

for si, seq in enumerate(seqs_lncRNA_ni):
    seqlen = len(seq)
    arr = np.chararray((seqlen,), buffer=seq)
    arr4 = np.chararray((CHARS_COUNT,), buffer=CHARS)
    for i, schar in enumerate(seq):
        for ii, char in enumerate(CHARS):
            if schar == char:
                if i<1500:
                    res_ni[si][i][ii] = 1



print res_ni

data_a_ni=res_ni


CHARS = 'ACGT'
CHARS_COUNT = len(CHARS)

maxlen = 1500#max(map(len, seqs))
res_gene_ni = np.zeros((len(seqs_mRNA_ni), maxlen, CHARS_COUNT), dtype=np.uint8)

for si, seq in enumerate(seqs_mRNA_ni):
    seqlen = len(seq)
    arr = np.chararray((seqlen,), buffer=seq)
    arr4 = np.chararray((CHARS_COUNT,), buffer=CHARS)
    for i, schar in enumerate(seq):
        for ii, char in enumerate(CHARS):
            if schar == char:
                if i<1500:
                    res_gene_ni[si][i][ii] = 1



print res_gene_ni
data_b_ni=res_gene_ni#[0:100]


labels_ni = np.asarray([[np.random.randint(0,1)] for p in range(0,len(seqs_lncRNA_ni))])



tweet_a_ni = Input(shape=(1500, 4))
tweet_b_ni = Input(shape=(1500, 4))


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


merged_name_ni=np.asarray(zip(lncRNA_name_list_ni, gene_name_list_ni))
merged_data_label_ni=np.hstack((intermediate_output_ni, labels_ni))
merged_dataset_ni=np.hstack((merged_name_ni, merged_data_label_ni))



import scipy.io as sio

io=np.vstack((intermediate_output, intermediate_output_ni))
lab=np.vstack((labels, labels_ni))
gene=gene_name_list+ gene_name_list_ni  
lnc=lncRNA_name_list+ lncRNA_name_list_ni

sio.savemat('./dataset/merged_lncRNA_mRNA.mat', {'x':{'data':io, 'nlab':lab,'ident':{'ident':lnc, 'milbag':gene}}})


