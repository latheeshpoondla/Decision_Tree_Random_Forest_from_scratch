# :deciduous_tree: Decision Tree and Random Forest :evergreen_tree::evergreen_tree: implementation from scratch
This project contains **from-scratch implementations** of both **Decision Tree Classifier** and **Random Forest Classifier** using only python, pandas, numpy with ML concepts like ensembling(bootstrapping and bagging) Trained and Tested with large datasets and compared performance with sci-kit learn.

## :books: Datasets
The model is tested with  
- Titanic dataset (891 x 12)
- Forest Cover Type real world large dataset (15120 x 56)

## Features

- 🧠 Decision Tree built using Gini Impurity
- 🌲 Random Forest using bagging and multiple trees
- 📊 Accuracy comparison with Scikit learn
- ⚙️ No external ML libraies (only, numpy + pandas)
- 🔣 Handles only numerical, categories can be encoded and trained


## 📏 Model Accuracy  

### Titanic Dataset
|Model  |Accuracy|
|------|-------|
|Custom Decision Tree|83.79%|
|Sklearn Decision Tree|83.24%|
|Custom Random Forest|85.47%|
|Sklearn Random Forest|85.47%|  

### Forest Cover Type
|Model |Accuracy|
|-----|-----|
|Custom Decision Tree|66.47%|
|Sklearn Decision Tree|78.87%|
|Custom Random Forest|61.18%|
|Sklearn Random Forest|86.94%|

## Future Enhancements
- :bar_chart: Visualization of Decision Tree

## Try it out
Clone the repository to try and modify
```bash
git clone https://github.com/latheeshpoondla/Decision_Tree_Random_Forest_from_scratch
cd Decision_Tree_Random_Forest_from_scratch
python Custom_DTree.py
```