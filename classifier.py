import io
import keras
import spacy
import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.layers import Add
from keras.utils import to_categorical
import keras.backend as K
from keras.preprocessing.text import one_hot
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense, Activation , Conv1D, MaxPooling1D, Bidirectional, Reshape, Flatten
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Model
from keras.layers import Concatenate, Input, Dense
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import svm


# Open the data: We have the polarity, the aspect term, the aspect category (AMBIENCE#GENERAL)
# and the review
train_set = pd.read_csv('/Users/.../data/traindata.csv', sep='\t', header= None)
dev_set = pd.read_csv('/Users/.../data/devdata.csv', sep='\t', header= None)


def aspect_term(df, colnum):
    """Only lower cases in the aspect term
    """

    lower_words = []
    for index, row in df.iterrows():
        lower_words.append(row[colnum].lower())
    df[colnum] = lower_words
    

def sentence_modifications(df, colnum):
    """Do some modifcations on the review to study it
    """

    lower_reviews = []
    # We only want lower cases in the review
    for index, row in df.iterrows():
        lower_reviews.append(row[colnum].lower())
    df[colnum] = lower_reviews


    # We use spacy to prepare text for deep learning
    nlp = spacy.load('en')
    
    cleaned_reviews = []
    # We want to remove stop words and punctuation:
    for doc in nlp.pipe(df[colnum].astype('unicode').values):
        if doc.is_parsed:
            cleaned_reviews.append(' '.join([tok.lemma_ for tok in doc if (not tok.is_stop and not tok.is_punct)])) 
            
        else:
            # We don't want an element of the list to be empty
            cleaned_reviews.append('') 
            
    df[colnum] = cleaned_reviews


    print("Sentence modifications done ")

def prepare_for_encoding(documents, number_of_words):
    """ We want to prepare the reviews to be prepared for embedding"""
    
    encoded_docs = [one_hot(d, number_of_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ') for d in documents]
    max_length = 100
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print("Reviews prepared for beeing embedded ! Shape: ")
    print(padded_docs.shape)
    
    return padded_docs


class Classifier:
    """
    
    The Classifier
    
    """

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        
        # First, we will clean the training set:
        train_set = pd.read_csv(trainfile, sep='\t', header= None)
        # To download the dataset below, :
        # https://www.kaggle.com/c/restaurant-reviews/data
        reviews_training = pd.read_csv('/Users/.../restaurant-train.csv', sep='\t', header= None)
        reviews_training = pd.concat([train_set[4],reviews_training[1]], axis = 0)
        
        # Do the cleaning: Lower cases, no stop word, no punctuation:
        aspect_term(train_set, 2)
        sentence_modifications(train_set, 4)
        
        # First, we create a tokenizer:
        voc_size = 7000
        tokenizer = Tokenizer(num_words = voc_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
        # Give the list of texts to train our tokenizer
        tokenizer.fit_on_texts(reviews_training)
        
        # Then, we save the existing tokenizer to apply it on new data.
        with open('tokenizer.pickle2', 'wb') as handle:
            	pickle.dump(tokenizer, handle)

        
        # Transform the reviews and the aspect terms to matrix of size len(train_set[4] / voc_size)
        review_matrix = pd.DataFrame(tokenizer.texts_to_matrix(train_set[4]))
        aspect_matrix = pd.DataFrame(tokenizer.texts_to_matrix(train_set[2]))
        
        
        #represent categorical values as vectors with one-hot encoding:
        category_encoder = LabelEncoder()
        transform_category = category_encoder.fit_transform(train_set[1])
        onehot_category = pd.DataFrame(to_categorical(transform_category))
        
        polarity_encoder = LabelEncoder()
        transform_polarity = polarity_encoder.fit_transform(train_set[0])
        onehot_polarity = pd.DataFrame(to_categorical(transform_polarity))
        
        
        # Define predictors and dependant variable
        X_train = pd.concat([onehot_category, aspect_matrix, review_matrix], axis=1)
        labels = onehot_polarity
                        
        # Multi-headed NN 
        left_branch = Input((7000, ))
        left_branch_dense = Dense(512, activation = 'relu')(left_branch)
        
        right_branch = Input((14012, ))
        right_branch_dense = Dense(512, activation = 'relu')(right_branch)
        merged = Concatenate()([left_branch_dense, right_branch_dense])
        output_layer = Dense(3, activation = 'softmax')(merged)
      
        model = Model(inputs=[left_branch, right_branch], outputs=output_layer)
        #optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit([np.array(review_matrix), np.array(X_train)], labels,epochs=2, verbose=1)
        # Store the model
        model.save('model.merged') 
        
        # Save the fitted label encoder for prediction decoding
        with open('labels.pickle', 'wb') as handle:
            	pickle.dump(polarity_encoder, handle)
            
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
        test_set = pd.read_csv(datafile, sep='\t', header= None)

        # Do the cleaning: Lower cases, no stop word, no punctuation:
        aspect_term(test_set, 2)
        sentence_modifications(test_set, 4)

        # Apply tokenizer on new data
        with open('tokenizer.pickle2', 'rb') as handle:
            	tokenizer = pickle.load(handle)
        
        # Define the BoW vectors using the same matrix
        tokenized_reviews = pd.DataFrame(tokenizer.texts_to_matrix(test_set[4]))
        tokenized_aspects = pd.DataFrame(tokenizer.texts_to_matrix(test_set[2]))

        #represent categorical values as vectors with one-hot encoding:
        category_encoder = LabelEncoder()
        transform_category = category_encoder.fit_transform(test_set[1])
        onehot_category = pd.DataFrame(to_categorical(transform_category))
        
        X_train = pd.concat([onehot_category, tokenized_aspects, tokenized_reviews], axis=1)
        
        # Load the model weights and architecture, predict for new data
        merged_model = load_model('model.merged')        
        predictions = merged_model.predict([np.array(X_train.iloc[:,7012:14012]),np.array(X_train)])
        predictions = predictions.argmax(axis=-1)  
        
        # Inverse the labels transformation
        with open('labels.pickle', 'rb') as handle:
            polarity_labels = pickle.load(handle)
        
        return polarity_labels.inverse_transform(predictions)
