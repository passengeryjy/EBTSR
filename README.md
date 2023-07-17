# ESGR
Code for paper "Evidence Sentence Graph Reasoning with Heuristic Strategies for Document-Level Biomedical Relation Extraction".
# DataSet
We use two widely used biomedical relation extraction datasets CDR and GDA to evaluate our model.
# Project Structure
The expected structure of files is:
```
ESGR
 |-- dataset
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |    |-- convert_CDR
 |    |    |    |-- convert_train.json
 |    |    |    |-- convert_test.json
 |    |    |    |-- convert_dev.json
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |    |    |-- convert_GDA
 |    |    |    |-- convert_train.json
 |    |    |    |-- convert_test.json
 |    |    |    |-- convert_dev.json
 |-- saved_model
      |-- best.model
      |-- log.txt
 |-- biobert_base
 |-- bioformer_base
 |-- scibert_base
 |-- convert_pro.py
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
