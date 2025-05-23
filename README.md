# Phishing Detection Using Graph Neural Networks (GNNs)

This repository presents a complete workflow for phishing detection leveraging **GraphSAGE**, a type of Graph Neural Network (GNN), with temporal modeling, causal sampling, and robustness testing.

## 🧠 Overview

Phishing attacks often involve subtle patterns that can be better detected using relational and temporal data. This project converts phishing datasets into graphs and applies a GNN model that:

- Respects **causal constraints** in message passing.
- Incorporates **temporal windowing** for realistic data flow.
- Tests **robustness** through noise injection.

## 🛠 Tech Stack

- **Programming Language:** Python
- **Graph Processing:** [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- **Machine Learning:** PyTorch, Scikit-learn
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib

## 📊 Workflow Summary

### 1. **Data Preprocessing**
- Load and clean phishing data from `phish.xlsx`
- One-hot encode categorical features
- Scale numerical features
- Combine features for each URL

### 2. **Graph Construction**
- Create a similarity graph using cosine similarity
- Connect each node to k=5 nearest neighbors
- Partition data into time windows of 10 samples
- Generate PyG `Data` objects for each time window

### 3. **Causal GraphSAGE Model**
- Custom model using `SAGEConv`, `BatchNorm`, `Dropout`
- Enforces **causal message passing** (no future info leakage)

### 4. **Noise Injection for Robustness**
- Add Gaussian noise to node features
- Randomly flip labels to simulate real-world inconsistencies

### 5. **Training**
- Trained with Binary Cross-Entropy loss and Adam optimizer
- Evaluated using AUC-ROC score and ROC curve visualization

## 📈 Evaluation

The model achieved strong performance on phishing detection:

| Metric     | Value  |
|------------|--------|
| Accuracy   | 86.36% |
| Precision  | 86.32% |
| Recall     | 86.36% |
| F1-Score   | 86.14% |
| AUC-ROC    | 0.9023 | Visualized in final plot |

# Visualizations
* Training Loss and Accuracy Over Epochs (Causal GraphSAGE): Visualizes the convergence of the model during causal training, showing decreasing loss and increasing accuracy over epochs. 
* Confusion Matrix: Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives from the final evaluation, illustrating the model's classification accuracy for each class. 
* ROC Curve: Illustrates the model's trade-off between True Positive Rate and False Positive Rate across various classification thresholds, with the AUC-ROC score quantifying overall performance. *
* Training Loss - Phishing Noise Training: Depicts the loss reduction during the training phase where noise was intentionally injected, demonstrating the model's ability to learn effectively despite data imperfections. *
* Overall Training Loss/Accuracy: Shows the general learning progression of the model, likely from an initial training phase, with loss decreasing and accuracy increasing. *

# Dependencies
The project relies on the following key libraries:

Python 3.x
torch (PyTorch)
torch-geometric (PyG)
torch-scatter
pandas
numpy
scikit-learn
matplotlib
```bash
git clone https://github.com/spk-22/Phish-Guard
```
```bash
pip install -r requirements.txt
# (Or manually install: torch, torch-geometric, scikit-learn, pandas, numpy, matplotlib)
# Ensure torch-geometric, torch-scatter, and torch-sparse versions are compatible with your PyTorch version.
```
``bash 
python phish.py
```
## 🔍 Use Case

This pipeline is ideal for cybersecurity researchers and engineers looking to detect phishing attempts using relational and temporal patterns within data.
The AUC-ROC score of 0.9023 signifies excellent discriminative power, even when trained on noisy data, indicating the model's strong ability to differentiate between phishing and legitimate attempts.
