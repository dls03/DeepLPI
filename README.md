# DeepLPI

Long   non-coding   RNAs   (lncRNAs)   regulate   diversebiological   processes   via   interactions   with   proteins.   Theexperimental   methods   to   identify   these   interactions   areexpensive and time-consuming, and thus many computationalmethods  have  been  proposed  to  predict  the  interactions.Most  of  these  computational  methods  are  based  on  knownlncRNA-protein  interactions  to  predict  new  interactions  byusing the information of sequences, secondary structures andexpression  profiles  of  lncRNAs  and  protein-coding  genes.Cross-validations  showed  that  these  methods  have  achievedpromising  prediction  performance.  However,  they  neglectedthe fact that a gene may encode multiple protein isoforms anddifferent isoforms of the same gene may interact differentlywith the same lncRNA.In  this  study,  we  propose  a  novel  method,  DeepLPI,to  predict  the  interactions  between  lncRNAs  and  proteinisoforms.   Our   method   uses   sequence   and   structure   datato  extract  intrinsic  features  such  as  functional  motifs,  anduses  expression  data  to  extract  topological  features  of  co-expression relationships. To combine these different data, weadopt a hybrid framework by integrating a multimodal deepleaning neural network (MDLNN) and a conditional randomfield  (CRF).  To  overcome  the  lack  of  known  interactionsbetween lncRNAs and protein isoforms, we apply a multipleinstance  learning  (MIL)  approach.  Our  method  iterativelytrains the MDLNN and CRF to predict interactions betweenlncRNAs  and  protein  isoforms  as  well  as  the  interactionsbetween lncRNAs and proteins (i.e., at the gene level).


Recommendation: 
Use python 2.7.5

\b Python Scripts:
deepLPI.py
- Main file 
- Input: sequence, structure and expression data
- Output: performance in terms of AUC and AUPRC

lncRNA_mRNA.py
- Process and merge lncRNA and isoform sequence data

lncRNA_mRNA_struct.py
- Process and merge lncRNA and isoform structure data

crf.py
- Conditional random field applying script

mil_nets/
- Codes from MINN with modifications to use in deepLPI project --  # Revisiting Multiple Instance Neural Networks, By [Xinggang Wang](http://mclab.eic.hust.edu.cn/~xwang/index.htm)
- convert the dataset to use as input of MIL framework
- codings for mil framework


\b Inputs:
dataset/
- contain the merged data of lncRNA and isoforms

lncRNA_expression_data.txt
- lncRNA expression data small version

refseqid_isoform_expression.txt
- isoform expression data small version


How to run:
python deepLPI.py --dataset merged_lncRNA_protein --dataset_struct merged_struct_lncRNA_protein





