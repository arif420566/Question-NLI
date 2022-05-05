#!/usr/bin/env python
# coding: utf-8

# ### **Import required packages**

# In[112]:


import pandas as pd
import numpy as np
import nltk
import re
from nltk import sent_tokenize, word_tokenize, WordPunctTokenizer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import validation_curve, learning_curve
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


# ### **Load the data**

# In[113]:


## Load the train, test and dev tsv files

## Read the train tsv
train_df = pd.read_csv(r'C:\Users\arif4\Downloads\Compressed\QNLI\train.tsv',sep='\t', error_bad_lines=False)

## Read the test tsv
test_df = pd.read_csv(r'C:\Users\arif4\Downloads\Compressed\QNLI\test.tsv',sep='\t', error_bad_lines=False)

## Read the dev tsv
dev_df = pd.read_csv(r'C:\Users\arif4\Downloads\Compressed\QNLI\dev.tsv',sep='\t', error_bad_lines=False)


# In[114]:


## Displaying the train dataset

train_df.head()


# In[115]:


## Displaying the test dataset

test_df.head()


# In[116]:


## Displaying the dev dataset

dev_df.head()


# ### Shape of the data

# In[117]:


# Shape of the data

print(f"Total train samples : {train_df.shape[0]}")
print(f"Total test samples : {test_df.shape[0]}")
print(f"Total validation or dev samples: {dev_df.shape[0]}")


# ### **Preprocessing**

# In[118]:


# Let us check null entries in our train data.
print("Number of missing values")
print(train_df.isnull().sum())


# Distribution of our training targets.

# In[119]:


print("Train Target Distribution")
print(train_df.label.value_counts())


# Distribution of our validation targets

# In[120]:


print("Validation Target Distribution")
print(dev_df.label.value_counts())


# In[121]:


## Target Label encoding
train_df['label'] = train_df['label'].apply(lambda x: 1 if x=='entailment' else 0)
dev_df['label'] = dev_df['label'].apply(lambda x: 1 if x=='entailment' else 0) 


# In[122]:


## We are considering the 20% of original data

train_df = train_df[0:round(len(train_df)*0.2)]
train_df.head()


# In[123]:


## Distribution of target classes
train_df.label.value_counts()


# ### **Normalization of Data**

# In[124]:


## Downloading the stopwords,punkt and wordnet

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[125]:


# stop words
stop_words = nltk.corpus.stopwords.words('english')
## Word Net Lemmatizer
wnl = WordNetLemmatizer()
def normalize_text(text):
    # lowercase and remove special characters\whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    # tokenize document
    tokens = word_tokenize(text)
    # Lemmatization of text
    lemma_tokens = [wnl.lemmatize(token,'v') for token in tokens]
    # filter stopwords out of document
    filtered_tokens = [token for token in lemma_tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_text)


# In[126]:


## Normalizing the train sentence and question features

norm_sentence_train = normalize_corpus(train_df['sentence'])
norm_question_train = normalize_corpus(train_df['question'])

## Normalizing the dev sentence and question features

norm_sentence_dev = normalize_corpus(dev_df['sentence'])
norm_question_dev = normalize_corpus(dev_df['question'])

## Normalizing the test sentence and question features

norm_sentence_test = normalize_corpus(test_df['sentence'])
norm_question_test = normalize_corpus(test_df['question'])


# ## **TFIDF Vectorizer for feature extraction**

# In[128]:


## Tfidf vectorizer for train sentence feature extraction

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(2, 2))

tfidf_sentence_train = tfidf.fit_transform(np.array(norm_sentence_train))


# In[129]:


## Shape of tfidf sentence features

tfidf_sentence_train.shape


# In[130]:


## Tfidf vectorizer for train question feature extraction

tfidf_question_train = tfidf.fit_transform(np.array(norm_question_train))

## Shape of tfidf question features

tfidf_question_train.shape


# In[131]:


## Tfidf vectorizer for dev sentence feature extraction

tfidf_sentence_dev = tfidf.fit_transform(norm_sentence_dev)

## Shape of tfidf sentence features

tfidf_sentence_dev.shape


# In[132]:


## Tfidf vectorizer for dev question feature extraction

tfidf_question_dev = tfidf.fit_transform(norm_question_dev)

## Shape of tfidf question features

tfidf_question_dev.shape


# In[133]:


## concatenating the tfidf_sentence_train and tfidf_question_train features

tfidf_train = np.concatenate((tfidf_sentence_train.toarray(), tfidf_question_train.toarray()), axis=1)


# In[134]:



## Shape of tfidf train features

tfidf_train.shape


# In[135]:


## concatenating the tfidf_sentence_dev and tfidf_question_dev features

tfidf_dev = np.concatenate((tfidf_sentence_dev.toarray(), tfidf_question_dev.toarray()), axis=1)


# In[136]:


## Shape of tfidf dev features

tfidf_dev.shape


# In[137]:


## Tfidf vectorizer for text sentence feature extraction

tfidf_sentence_test = tfidf.fit_transform(norm_sentence_test)

## Shape of tfidf test sentence features

tfidf_sentence_test.shape


# In[138]:


## Tfidf vectorizer for text question feature extraction

tfidf_question_test = tfidf.fit_transform(norm_question_test)

## Shape of tfidf test question features

tfidf_question_test.shape


# In[139]:


## concatenating the tfidf_sentence_test and tfidf_question_test features

tfidf_test = np.concatenate((tfidf_sentence_test.toarray(), tfidf_question_test.toarray()), axis=1)


# In[140]:


## Shape of tfidf test features

tfidf_test.shape


# ### **Building the Multinomial Naive Bayes as Base model**

# In[141]:


## Multinomial Naive Bayes model

nb = MultinomialNB()

nb = nb.fit(tfidf_train, train_df['label'])

nb


# ### Checking the performance on validation data

# In[142]:


## performance on dev data

nb_pred = nb.predict(tfidf_dev)


# In[143]:


## Predictions

nb_pred


# In[145]:


## dev accuracy score

nb_score = accuracy_score(dev_df['label'], nb_pred)
print(f'Accuracy score of Naive Bayes for dev data : {nb_score}')


# ### Predictions using test data

# In[146]:


## predicting labels using test data

nb_pred1 = nb.predict(tfidf_test)

## predictions
nb_pred1


# In[147]:


## Saved predictions in CSV file

test_df['labels'] = nb_pred1

test_df.to_csv('test_data.csv')


# ### Classification Report

# In[148]:


## Printing the classification report

nb_report = classification_report(dev_df['label'], nb_pred, target_names=['not_entailment', 'entailment'])

print(nb_report)


# ### Confusion Matrix

# In[149]:


## Confusion Matrix

nb_cm = confusion_matrix(dev_df['label'], nb_pred, labels=[1,0])

print(nb_cm)


# In[151]:


ConfusionMatrixDisplay.from_predictions(dev_df['label'], nb_pred)


# ### **Error Analysis**

# In[65]:


## Plot learning curve

train_sizes, train_scores, valid_scores = learning_curve(
nb, tfidf_train, train_df['label'], train_sizes=np.linspace(0.1, 1.0, 5), cv=5)


# In[66]:


train_sizes


# In[67]:


train_scores


# In[68]:


valid_scores


# In[69]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)


# In[70]:


# Plot learning curve

fig = plt.subplots(1, 1, figsize=(10, 15))

plt.title("Learning Curves (Naive Bayes)")
plt.xlabel("Training examples")
plt.ylabel("Score")

plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",)

plt.fill_between(
    train_sizes,
    valid_scores_mean - valid_scores_std,
    valid_scores_mean + valid_scores_std,
    alpha=0.1,
    color="g",)

plt.plot(
    train_sizes, train_scores_mean, "o-", color="r", label="Training score")
plt.plot(
    train_sizes, valid_scores_mean, "o-", color="g", label="Cross-validation score")
plt.legend(loc="best")


# In[44]:




