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
 |-- biobert_base
 |-- bioformer_base
 |-- scibert_base
 |-- utils.py
 |-- graph.py
 |-- prepro.py
 |-- long_seq.py
 |-- losses.py
 |-- run_cdr.py
 |-- run_gda.py
 |-- rgcn_utils.py
 |-- EBTSR.py
```
# Training
Models trained on CDA and GDA model with the following command:
```
>> python run_cdr.py  # for CDR
>> python run_gda.py  # for GDA
```
You can save the model by setting the ``--save_path`` argument before training. The model correponds to the best dev results will be saved in the best.model. 
# Evaluation
You can evaluate the saved model by setting the ``--load_path`` argument, the best model will be load on benchmarks.
