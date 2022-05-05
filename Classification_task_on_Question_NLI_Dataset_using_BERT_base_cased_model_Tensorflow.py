#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## install transformers

get_ipython().system('pip install transformers')


# In[ ]:


### import the required libraries

import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
import csv
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import TFBertModel, BertTokenizer

import warnings
warnings.filterwarnings('ignore')

transformers.logging.set_verbosity_error()


# ### **Load the data**
# 

# In[ ]:


## Load the train, test and dev tsv files

## Read the train tsv
train_df = pd.read_csv(r'C:\Users\arif4\Downloads\Compressed\QNLI\train.tsv',sep='\t', error_bad_lines=False)

## Read the test tsv
test_df = pd.read_csv(r'C:\Users\arif4\Downloads\Compressed\QNLI\test.tsv',sep='\t', error_bad_lines=False)

## Read the dev tsv
dev_df = pd.read_csv(r'C:\Users\arif4\Downloads\Compressed\QNLI\dev.tsv',sep='\t', error_bad_lines=False)


# In[ ]:


## Displaying the train dataset

train_df.head()


# In[ ]:


## Displaying the test dataset

test_df.head()


# In[ ]:


## Displaying the dev dataset

dev_df.head()


# ### Shape of the data

# In[ ]:


# Shape of the data

print(f"Total train samples : {train_df.shape[0]}")
print(f"Total test samples : {test_df.shape[0]}")
print(f"Total validation or dev samples: {dev_df.shape[0]}")


# ### **Preprocessing**

# In[ ]:


# Let us check null entries in our train data.
print("Number of missing values")
print(train_df.isnull().sum())


# Distribution of our training targets.

# In[ ]:


print("Train Target Distribution")
print(train_df.label.value_counts())


# Distribution of our validation targets.

# In[ ]:


print("Validation Target Distribution")
print(dev_df.label.value_counts())


# In[ ]:


## Target Label encoding
train_df['label'] = train_df['label'].apply(lambda x: 1 if x=='entailment' else 0)
dev_df['label'] = dev_df['label'].apply(lambda x: 1 if x=='entailment' else 0) 


# In[ ]:


y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=2)


# In[ ]:


y_train.shape


# In[ ]:


y_dev = tf.keras.utils.to_categorical(dev_df.label, num_classes=2)


# In[ ]:


y_dev.shape


# In[ ]:


## Considering the 0.05% of train data

train_df_new = train_df[0:round(len(train_df)*0.05)]
train_df_new.head()


# In[ ]:


## Distribution of target classes
train_df_new.label.value_counts()


# In[ ]:


y_train_new = tf.keras.utils.to_categorical(train_df_new.label, num_classes=2)


# In[ ]:


y_train_new.shape


# ### **Configuration**

# In[ ]:


max_length = 128  # Maximum length of input sentence to the model.
batch_size = 16
epochs = 5

# Labels in our dataset.
# 0 for not_entailment, 1 for entailment
labels = ["not_entailment","entailment"]


# ### **Keras Custom Data Generator**

# In[ ]:


class BERTDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of question and sentence input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased"
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
      # Retrieves the batch of index.
      indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
      sentence_pairs = self.sentence_pairs[indexes]

      # With BERT tokenizer's batch_encode_plus batch of both the sentences are
      # encoded together and separated by [SEP] token.
      encoded = self.tokenizer.batch_encode_plus(
          sentence_pairs.tolist(),
          add_special_tokens=True,
          max_length=max_length,
          return_attention_mask=True,
          return_token_type_ids=True,
          pad_to_max_length=True,
          truncation = True,
          return_tensors="tf"
      )

      # Convert batch of encoded features to numpy array.
      input_ids = np.array(encoded["input_ids"], dtype="int32")
      attention_masks = np.array(encoded["attention_mask"], dtype="int32")
      token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

      # Set to true if data generator is used for training/validation.
      if self.include_targets:
          labels = np.array(self.labels[indexes], dtype="int32")
          return [input_ids, attention_masks, token_type_ids], labels
      else:
          return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


# ## **Build the model**

# In[ ]:


# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = TFBertModel.from_pretrained("bert-base-cased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    #sequence_output, pooled_output = bert_model(
     #   input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, return_dict=False
    #)
    outputs = bert_model.bert(input_ids,attention_masks)

    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(outputs[0])
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.4)(concat)
    output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.005),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")
model.summary()


# Create train and validation data generators

# In[ ]:


train_data = BERTDataGenerator(
    train_df_new[["question", "sentence"]].values.astype("str"),
    y_train_new,
    batch_size=batch_size,
    shuffle=True,
)


# In[ ]:


dev_data = BERTDataGenerator(
    dev_df[["question", "sentence"]].values.astype("str"),
    y_dev,
    batch_size=batch_size,
    shuffle=True,
)


# ## **Train the Model**
# 
# Training is done only for the top layers to perform "feature extraction",
# which will allow the model to use the representations of the pretrained model.

# In[ ]:


history = model.fit(
    train_data,
    validation_data=dev_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1)


# In[ ]:


## saving the model

model.save('C:\Users\arif4\Downloads\Compressed\QNLI\model\new_model.h5', save_format="h5")


# In[ ]:


## predictions

bert_preds = model.predict(dev_data)
bert_preds


# In[ ]:


## predictions
preds = bert_preds.argmax(axis=1)
preds


# In[ ]:


preds.shape


# In[ ]:


y_true = [dev_data[i][-1].argmax(1) for i in range(len(dev_data))]


# In[ ]:


y_true_new = np.concatenate(y_true, axis=0)


# In[ ]:


y_true_new.shape


# ### Classification Report

# In[ ]:


## Printing the classification report

bert_report = classification_report(y_true_new, preds, target_names=['not_entailment', 'entailment'])

print(bert_report)


# ### Confusion Matrix

# In[ ]:


## Confusion Matrix

bert_cm = confusion_matrix(y_true_new, preds, labels=[1,0])

print(bert_cm)


# ### Plot Confusion Matrix

# In[ ]:


## Plotting the confusion matrix

ConfusionMatrixDisplay.from_predictions(y_true_new, preds)


# ### **Error Analysis**

# In[ ]:


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['acc']
val_accuracy = history_dict['val_acc']

epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the model accuracy vs Epochs

ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
ax[0].set_title('Training & Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuracy', fontsize=16)
ax[0].legend()
# Plot the loss vs Epochs
ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()


# In[ ]:




