# Human Language Technologies
## Academic Year Project 2023/2024

# Cyberbullying Classification

## Introduction

This project, conducted as part of the Human Language Technologies (HLT) course, aims to develop and evaluate a Natural Language Processing (NLP) model for the classification of tweets from the social media platform X (formerly known as Twitter) as potential acts of cyberbullying or offensive behavior.

## Motivations

The project is driven by three primary motivations:

1. **Application of Knowledge**: To apply the theoretical and methodological concepts learned during the HLT course.
2. **Social Relevance**: To address the growing social and psychological issue of cyberbullying, which has become more prevalent since the Covid-19 pandemic.
3. **Challenging Goals**: To meet the challenging goals set by the authors of the dataset and to contribute meaningful insights into the domain of cyberbullying detection.

## Project Structure

The project is organized into several key directories:

- **`_chunckdevs`**: A custom library developed by the team specifically for this project.
- **`data`**: Contains all datasets used for training and evaluation.
- **`notebooks`**: Includes commented Jupyter notebooks for preprocessing, baseline models, advanced models, and transformer-based models.
- **`outputs`**: Stores generated outputs, including trained models and other relevant files.
- **`requirements.txt`**: Contains all the libraries needed for code execution.

## Dataset and Goal

The dataset, sourced from Kaggle, consists of over 47,000 tweets, each labeled according to the type of cyberbullying. The dataset is balanced, with each class containing approximately 8,000 labels. Tweets are categorized either as descriptions of bullying events or as the bullying acts themselves. The primary objectives are:

1. **Binary Classification**: To identify whether a tweet constitutes an act of cyberbullying or not.
2. **Multiclass Classification**: To detect the specific type of discriminatory act, with labels including:
   - Age
   - Ethnicity
   - Gender
   - Religion
   - Other types of cyberbullying
   - Not cyberbullying

## Data Understanding and Preparation

### Data Understanding

Initial exploration included the creation of word clouds for each class, revealing significant semantic differences related to cyberbullying. Hashtags, initially considered, were eventually excluded due to their low frequency and lack of specificity.

### Data Preprocessing

Two versions of the dataset were prepared: one containing all tweets and another with only English texts. Both versions were split into development and test sets. Normalization was applied exclusively to the development set tweets. Duplicate tweets, particularly those labeled as "other cyberbullying," were identified and removed.

## Classification

### Models Implemented

A variety of models were implemented and evaluated:

- Baseline models
- Advanced models
- Transformer-based models
- Ensemble models

### Feature Engineering

Features were engineered for both baseline and advanced models, with extensive hyperparameter tuning to optimize performance.

### Evaluation Metrics

Model performance was evaluated using metrics such as precision, recall, and F1-score.

## Results for Classification

### Baseline and Advanced Models

Ensemble models achieved the highest F1-scores, although precision for certain classes remained challenging.

### Comparison with State-of-the-Art

Our models were benchmarked against state-of-the-art (SOTA) models to evaluate relative performance.

## Conclusions

Our analysis demonstrates that while machine learning models can effectively distinguish between different types of cyberbullying, they struggle with context and intent, particularly in distinguishing non-cyberbullying tweets from harmful messages. This underscores the need for further research into context disambiguation and intent understanding to improve the efficacy of cyberbullying detection models.
