# SPMN 

SPMN module of SPFlow library implements the structure learning algorithm for Sum-Product-Max Networks(**SPMN**) which generalise Sum-Product Networks(**SPN**) for the class of decison-making problems.

## Getting Started

See https://github.com/SPFlow/SPFlow for installation instructions for SPFlow library

## Using SPMN Module

Look at *spmn/data* folder for a list of sample datasets to use with spmn structure learning algorithm. *spmn/meta_data* contains information about *partial order, decision nodes, utility node* for each of the data sets.
```python
    import pandas as pd    
    csv_path = "Dataset5/Computer_diagnostician.tsv"
    df = pd.DataFrame.from_csv(csv_path, sep='\t')
 ```
Provide *partial order, decision nodes, utility node* for the dataset
```python
    partial_order = [['System_State'], ['Rework_Decision'],
                     ['Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 'Rework_Cost']]
    utility_node = ['Rework_Cost']
    decision_nodes = ['Rework_Decision']
```
var_set is list of all variables in sequence of partial order excluding decison variables
```python
    var_set = [var for var_set in partial_order for var in var_set]
    for d in decision_nodes:
        var_set.remove(d)
 
     #var_set = ['System_State','Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 'Rework_Cost' ]
```
Pre-process data 
```python
    from spn.algorithms.SPMNDataUtil import align_data
    import numpy as np
    
    df1, column_titles = align_data(df, partial_order)  #aligns data in partial order sequence
    col_ind = column_titles.index(utility_node[0]) 
    
    df_without_utility = df1.drop(df1.columns[col_ind], axis=1)
    from sklearn.preprocessing import LabelEncoder
    df_without_utility_categorical = df_without_utility.apply(
        LabelEncoder().fit_transform)  # transform categorical string values to categorical numerical values
    df_utility = df1.iloc[:, col_ind]
    df = pd.concat([df_without_utility_categorical, df_utility], axis=1, sort=False)

    train_data = df.values
```
```python
    from spn.algorithms.SPMN import learn_spmn
    spmn = SPMN.learn_spmn(train_data , partial_order , decision_nodes, utility_node , var_set,
                   util_to_bin = False )
                 
    from spn.io.Graphics import plot_spn
    plot_spn(spmn, "ds6.pdf", feature_labels=['SS', 'LBF', 'IBF', 'RO', 'RC'])
```    
    
We can calculate maximum expected utility of test data and return the best decision at each decision node
```python
    from spn.algorithms.MEU import meu
    
    #test data - only include random variables and utility variable, exclude decision variables
    test_data = np.array([[  0., 1., 0., 1., 175.],[  0., 1., 1., 0., 300.]]) 
    meu, decisions = meu(spmn, test_data1.astype(float) ))
```    
The output for meu and decisions is:

    [0.69962597, 0.07494787]
    {'Rework_Decision': array([[0, 1],[1, 0]])}  #decision for 0th instance is '1' and 1st instance is '0'
    
We can convert utility variable to binary random variable using cooper transformation
```python  
    from spn.algorithms.SPMNDataUtil import cooper_tranformation
    bin_data = cooper_tranformation(train_data, col_ind)   #col_ind is index of utility variable in train data
    spmn = learn_spmn(bin_data , partial_order , decision_nodes, utility_node , var_set,
                   util_to_bin = True )
```
## Papers implemented
* Mazen Melibari, Pascal Poupart, Prashant Doshi. "Sum-Product-Max Networks for Tractable Decision Making". In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, 2016.

### Limitations
* works with categorical variables and one utility node. Utility node can be real valued
    

