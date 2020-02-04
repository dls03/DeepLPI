# DeepLPI

DeepLPI: a multimodal deep learning method for predicting the interactions between lncRNAs and protein isoforms

Long   non-coding   RNAs   (lncRNAs)   regulate   diversebiological   processes   via   interactions   with   proteins.   Theexperimental   methods   to   identify   these   interactions   areexpensive and time-consuming, and thus many computationalmethods  have  been  proposed  to  predict  the  interactions.Most  of  these  computational  methods  are  based  on  knownlncRNA-protein  interactions  to  predict  new  interactions  byusing the information of sequences, secondary structures andexpression  profiles  of  lncRNAs  and  protein-coding  genes.Cross-validations  showed  that  these  methods  have  achievedpromising  prediction  performance.  However,  they  neglectedthe fact that a gene may encode multiple protein isoforms anddifferent isoforms of the same gene may interact differentlywith the same lncRNA.In  this  study,  we  propose  a  novel  method,  DeepLPI,to  predict  the  interactions  between  lncRNAs  and  proteinisoforms.   Our   method   uses   sequence   and   structure   datato  extract  intrinsic  features  such  as  functional  motifs,  anduses  expression  data  to  extract  topological  features  of  co-expression relationships. To combine these different data, weadopt a hybrid framework by integrating a multimodal deepleaning neural network (MDLNN) and a conditional randomfield  (CRF).  To  overcome  the  lack  of  known  interactionsbetween lncRNAs and protein isoforms, we apply a multipleinstance  learning  (MIL)  approach.  Our  method  iterativelytrains the MDLNN and CRF to predict interactions betweenlncRNAs  and  protein  isoforms  as  well  as  the  interactionsbetween lncRNAs and proteins (i.e., at the gene level).


### Installation and recommendation: 
- Install [Keras](https://keras.io/) with [TensorFlow](https://keras.io/backend/) backend.
- Use python 2.7.5

### Python Scripts:
deepLPI.py
- Model file

lncRNA_mRNA.py
- Process and merge lncRNA and isoform sequence data.

lncRNA_mRNA_struct.py
- Process and merge lncRNA and isoform structure data.

crf.py
- Conditional random field optimization tool initially developed in [DIFFUSE: predicting isoform functions from sequences and expression profiles via deep learning](https://doi.org/10.1093/bioinformatics/btz367) 

mil_nets/
- Codes from MINN with modifications to use in deepLPI project : [Revisiting Multiple Instance Neural Networks](http://mclab.eic.hust.edu.cn/~xwang/index.htm), By Xinggang Wang
- Convert the dataset to use as input of MIL framework
- Scripts for MIL learning


### Data:
dataset/
- Contain the merged data of lncRNAs and proteins.

lncRNA_expression_data.txt
- lncRNA expression data small version.

refseqid_isoform_expression.txt
- isoform expression data small version.


### Run deepLPI:
python deepLPI.py --dataset merged_lncRNA_protein --dataset_struct merged_struct_lncRNA_protein





