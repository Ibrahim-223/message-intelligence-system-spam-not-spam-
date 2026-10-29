# Message Intelligence System — Spam vs Legitimate Message Classification

## 📌 Project Overview

The **Message Intelligence System** is a Machine Learning project that classifies messages into two categories: **Spam (1)** and **Legitimate (0)**.

The project implements three classification algorithms — K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Naive Bayes — and compares their performance using standard classification metrics.

## 🎯 Objectives

- Understand conditional probability and Bayes' Theorem.
- Prepare and preprocess the dataset for machine learning.
- Implement KNN with different K values and distance metrics.
- Implement SVM using Linear and RBF kernels.
- Implement the Gaussian Naive Bayes classifier.
- Compare models using Accuracy, Precision, Recall, and F1-score.
- Identify suitable classifiers for spam detection.

## 📂 Dataset Information

The dataset contains **5,200 messages** and the following features:

| Feature | Description |
|---|---|
| `message_id` | Unique message identifier |
| `message_text` | Original message text |
| `message_length` | Length of the message |
| `word_count` | Number of words |
| `num_urls` | Number of URLs |
| `num_digits` | Number of digits |
| `num_special_chars` | Number of special characters |
| `spam_keyword_score` | Spam keyword score |
| `legit_keyword_score` | Legitimate keyword score |
| `sender_activity_score` | Sender activity score |
| `sender_account_age_days` | Sender account age in days |
| `messages_sent_last_24h` | Messages sent in the last 24 hours |
| `hour_of_day` | Hour when the message was sent |
| `day_of_week` | Day of the week |
| `spam_label` | Target class: 0 = Legitimate, 1 = Spam |

### Class Distribution

- Legitimate messages (0): 4,227
- Spam messages (1): 973
- Total messages: 5,200

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib and Seaborn (for visualizations, if used)
- Jupyter Notebook

## ⚙️ Project Workflow

1. Load and explore the dataset.
2. Check missing values, duplicates, and class distribution.
3. Remove the unique message identifier and prepare numerical features.
4. Split the dataset into training and testing sets using an 80:20 split.
5. Scale the numerical features.
6. Apply SMOTE to the training data to balance the classes.
7. Train KNN, SVM, and Naive Bayes classifiers.
8. Evaluate each model on the test dataset.
9. Compare the evaluation metrics and summarize the results.

**Train-test split:** 4,160 training samples and 1,040 testing samples.

**SMOTE:** The resampled training dataset contains 3,382 samples per class, or 6,764 samples in total. The test dataset retains its original class distribution.

## 🤖 Machine Learning Algorithms

### 1. K-Nearest Neighbors (KNN)

KNN classifies a message based on the majority class among its nearest data points.

Experiments include:
- Different K values: 3, 5, 7, 9, and 11.
- Distance metrics: Euclidean, Manhattan, and Minkowski.
- Confusion matrix and incorrect prediction analysis.

**Strength:** Simple and intuitive.

**Limitation:** Prediction can become slow on large datasets.

### 2. Support Vector Machine (SVM)

SVM finds a decision boundary that separates the classes with a large margin.

Kernels implemented:
- **Linear kernel:** Uses a linear decision boundary.
- **RBF kernel:** Handles nonlinear decision boundaries.

**Strength:** Can perform well on complex classification problems.

**Limitation:** Can require more computational resources and is harder to interpret.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes predicts classes using Bayes' Theorem and assumes that numerical features follow a Gaussian distribution within each class, along with conditional independence of features.

**Strength:** Fast and provides class probabilities.

**Limitation:** Its assumptions may not accurately represent real-world feature relationships.

## 📐 Probability Concepts

### Conditional Probability

Conditional probability measures the probability of an event given that another event has occurred.

### Bayes' Theorem

Bayes' Theorem updates the probability of a class using observed evidence. It is the mathematical foundation of the Naive Bayes classifier.

### Class Probability Prediction

The `predict_proba()` method is used to obtain the estimated probability of each class for a sample.

## 📊 Evaluation Metrics

All models are evaluated using the following metrics:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted spam messages that are actually spam.
- **Recall:** Proportion of actual spam messages correctly detected.
- **F1-score:** Harmonic mean of precision and recall.

These metrics help evaluate both overall performance and the model's ability to detect spam without incorrectly flagging legitimate messages.

## 📈 Results and Model Comparison

During the experiments, KNN achieved an accuracy and F1-score of 1.00 on the test dataset. The initial comparisons also showed perfect scores for the evaluated models.

However, these results require further validation because perfect classification scores can indicate data leakage, duplicate patterns, or features that strongly reveal the target label.

The final model should be selected after checking the data pipeline and evaluating performance on genuinely unseen messages.

## 💼 Business Recommendation

For real-world spam detection, precision and recall should both be considered.

- High precision helps prevent legitimate messages from being incorrectly marked as spam.
- High recall helps detect more spam messages.
- F1-score helps balance precision and recall.

Naive Bayes can serve as a fast baseline, while KNN and SVM can be compared for classification performance. Before deployment, the chosen model should be tested on fresh messages and monitored for incorrect predictions.

## 🏁 Conclusion

This project demonstrates the implementation and comparison of KNN, SVM, and Naive Bayes for spam message classification. It also applies probability concepts, class balancing, feature scaling, and standard evaluation metrics.

The project provides practical experience with the machine learning classification workflow and highlights the importance of reliable evaluation before real-world deployment.

## 👨‍💻 Project Status

**Status:** Completed — initial model implementation and evaluation.

**Future Improvements:**
- Validate results using a leakage-free pipeline.
- Test on new, unseen messages.
- Explore text-based features using TF-IDF.
- Tune model hyperparameters.
- Monitor precision, recall, and F1-score on real-world data.
