#!/bin/bash -l

# Argument explanation:
# --dataset: merged sequence data of lncRNA and protein
# --dataset_struct: merged structure data of lncRNA and protein
# --interaction: Querying about the  interaction between lncRNA and protein should be give in the form of lncRNA-protein
# --pre_trained_weight: pre-trained weight of the model (current default: model_deepLPI.h5)
# output will be displayed in console 


python deepLPI.py --dataset merged_lncRNA_protein_mini2 --dataset_struct merged_struct_lncRNA_protein_mini2 --interaction TUG1-BTG3
#python deepLPI.py --dataset merged_lncRNA_protein_mini2 --dataset_struct merged_struct_lncRNA_protein_mini2 --interaction TUG1-RAB3GAP2
#python deepLPI.py --dataset merged_lncRNA_protein_mini2 --dataset_struct merged_struct_lncRNA_protein_mini2 --interaction MYCNOS-GALNT12
#python deepLPI.py --dataset merged_lncRNA_protein_mini2 --dataset_struct merged_struct_lncRNA_protein_mini2 --interaction TUG1-TM9SF2


