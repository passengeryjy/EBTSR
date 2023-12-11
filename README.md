# EBTSR
Code for paper "Enhancing Document-Level Biomedical Relation Extraction through Evidence-Based Two-Stage Reasoning ".
# DataSet
We use two widely used biomedical relation extraction datasets CDR and GDA to evaluate our model.
# Project Structure
The expected structure of files is:
```
EBTSR
 |-- dataset
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |-- saved_model
      |-- best.model
      |-- log.txt
 |-- biobert_base
 |-- bioformer_base
 |-- scibert_base
 |-- utils.py
 |-- adj_utils.py
 |-- prepro.py
 |-- long_seq.py
 |-- losses.py
 |-- train_cdr.py
 |-- train_gda.py
 |-- rgcn.py
 |-- model.py
```
# Training and Evaluation
## Training
Train CDA and GDA model with the following command:
```
>> python train_cdr.py  # for CDR
>> python train_gda.py  # for GDA
```
You can save the model by setting the ``--save_path`` argument before training. The model correponds to the best dev results will be saved. 
# Evaluation
You can evaluate the saved model by setting the ``--load_path`` argument, then the code will skip training and evaluate the saved model on benchmarks.
