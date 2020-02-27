# DeepLPI

DeepLPI: a multimodal deep learning method for predicting the interactions between lncRNAs and protein isoforms

Long non-coding RNAs (lncRNAs) regulate diverse biological processes via interactions with proteins. The experimental methods to identify these interactions are expensive and time-consuming, and thus many computational methods have been proposed to predict the interactions. Most of these computational methods are based on known lncRNA-protein interactions to predict new interactions by using the information of sequences, secondary structures and expression profiles of lncRNAs and protein-coding genes. Cross-validations showed that these methods have achieved promising prediction performance. However, they neglected the fact that a gene may encode multiple protein isoforms and different isoforms of the same gene may interact differently with the same lncRNA. In this study, we propose a novel method, DeepLPI, to predict the interactions between lncRNAs and protein isoforms. Our method uses sequence and structure data to extract intrinsic features such as functional motifs, and uses expression data to extract topological features of co-expression relationships. To combine these different data, we adopt a hybrid framework by integrating a multimodal deep leaning neural network (MDLNN) and a conditional random field (CRF). To overcome the lack of known interactions between lncRNAs and protein isoforms, we apply a multiple instance learning (MIL) approach. Our method iteratively trains the MDLNN and CRF to predict interactions between lncRNAs and protein isoforms as well  as the interactions between lncRNAs and proteins (i.e., at the gene level).


### Dependencies: 
- [Python 2.7.5](https://www.python.org/downloads/release/python-275/)
- [Keras](https://keras.io/)
- [TensorFlow](https://keras.io/backend/)
- [MINNs](https://github.com/yanyongluan/MINNs)
- [SciPy](https://www.scipy.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)


### Data:
- dataset
  - The merged sequence data of lncRNAs and protein isoforms
  - The merged structural data of lncRNAs and protein isoforms
- lncRNA_expression_data.txt
  - A mini example of lncRNA expression data
- isoform_expression_data.txt
  - A mini example of isoform expression data
- The full datasets used in the paper can be found at [here](https://drive.google.com/drive/folders/1qWZq4pKoPJj6u_pbNgKAm_zodxpPxD9S?usp=sharing).


### Prediction using DeepLPI:
- Run the script in ./demo.sh to generate the prediction for a candidate interaction pair based on a pre-trained model. The first two arguments in the script can be found in the above datasets. You may replace the interaction pair in this script by another one.
- The output contains the predicted result for the queried interaction. 

### Training DeepLPI:
- Run the script in ./train.sh to train a new model. Again, the arguments can be found in the above datasets. You may replace the datasets in the script by others to train a different model.
- The output will report the average performance in terms of AUC and AUPRC, over 5-fold cross validation.







