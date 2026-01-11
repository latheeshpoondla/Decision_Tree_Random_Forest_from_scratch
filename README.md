# :deciduous_tree: Decision Tree & Random Forest :evergreen_tree: from Scratch (Python)
This repository contains from-scratch implementations of a Decision Tree Classifier and a Random Forest Classifier, built using only Python, NumPy, and Pandas, without relying on external machine learning libraries like scikit-learn.

The project demonstrates core ML concepts such as Gini Impurity, recursive tree building, bootstrapping, bagging, and Out-of-Bag (OOB) error estimation, and compares the results with scikit-learn models.

## 📌 Key Highlights
✅ Decision Tree built using **Gini Impurity**
🌲 Random Forest implemented using **Bootstrap Aggregation (Bagging)**
📦 **Out-of-Bag (OOB) error** calculation without cross-validation
📊 Performance comparison with **scikit-learn**
⚙️ No external ML libraries used (only NumPy + Pandas)
🔢 Supports **numerical features** (categorical features must be encoded)

## 🧠 Implemented Algorithms
### 1️⃣ Decision Tree Classifier
* Binary splits using **Gini Impurity**
* Recursive tree growth
* Supports:
  * max_depth
  * max_features (for Random Forest compatibility)
* Majority voting at leaf nodes

### 2️⃣ Random Forest Classifier
* Ensemble of custom Decision Trees
* Bootstrap sampling for each tree
* Feature sub-sampling (```sqrt(n_features)``` by default)
* **Out-of-Bag voting** for unbiased error estimation

## 📂 Project Structure
```
├── Classifier.py          # Base abstract classifier + Gini utilities
├── Custom_DTree.py        # Decision Tree implementation
├── Custom_RForest.py      # Random Forest implementation with OOB
├── dt_rf_notebook.ipynb   # Experiments and comparisons
├── README.md
```
## 📊 Datasets Used
The models were trained and evaluated on:
* **Titanic Dataset** (891 × 12)
* **Forest Cover Type Dataset** (15,120 × 56)
## 📈 Model Performance  
### 🚢 Titanic Dataset
|Model  |Accuracy|
|------|-------|
|Custom Decision Tree|86.03%|
|Sklearn Decision Tree|85.47%|
|Custom Random Forest|86.03%|
|Sklearn Random Forest|86.03%|  

### 🌲 Forest Cover Type Dataset
|Model |Accuracy|
|-----|-----|
|Custom Decision Tree|79.43%|
|Sklearn Decision Tree|80.13%|
|Custom Random Forest|83.66%|
|Sklearn Random Forest|87.53%|
|Custom Random Forest|0.174|
|Sklearn OOB Error|0.136|

⚠️ Performance gap highlights the importance of advanced optimizations used in production-grade libraries.
## 📐 Out-of-Bag (OOB) Error
The Random Forest implementation computes OOB error by:
1. Tracking samples not included in each tree’s bootstrap set
2. Aggregating OOB predictions via majority voting
3. Estimating generalization error without a validation set
```
rf.oob_error
```
This mimics scikit-learn's ```oob_score``` mechanism internally. 

## 🔮 Future Enhancements
* 📊 Decision Tree visualization (Graphviz / Matplotlib)
* ⚡ Performance optimizations
* 📦 Support for categorical features
* 📉 Feature importance calculation
* 🧪 Regression trees & Random Forest Regressor

## 🎯 Learning Outcomes
This project is ideal for:
* Understanding how tree-based models work internally
* Learning ensemble learning from first principles
* Academic coursework & ML interviews
* Building intuition beyond scikit-learn abstractions

## Try it out
Clone the repository to try and modify
```bash
git clone https://github.com/latheeshpoondla/Decision_Tree_Random_Forest_from_scratch
cd Decision_Tree_Random_Forest_from_scratch
python Custom_DTree.py
```
