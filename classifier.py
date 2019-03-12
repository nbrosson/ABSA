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
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense, Activation , Conv1D, MaxPooling1D, Bidirectional, Reshape, Flatten
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Concatenate, Input, Dense
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import svm


# Open the data: We have the polarity, the aspect term, the aspect category (AMBIENCE#GENERAL)
# and the review
train_set = pd.read_csv('/Users/.../traindata.csv', sep='\t', header= None)
dev_set = pd.read_csv('/Users/.../devdata.csv', sep='\t', header= None)


#### Data cleaning Functions ###

# As we want to do the data cleaning for every datasets (training, dev and test),
# we will do data cleaning functions that we will put both in the train and the predict methods

def aspect_term(df, colnum):
    """Only lower cases in the aspect term
    """

    lower_words = []
    for index, row in df.iterrows():
        lower_words.append(row[colnum].lower())
    df[colnum] = lower_words
    


# To make the best out of our data, we will also study the entire sentences. 
# Thus, we will clean it:

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


def prepare_for_encoding(documents, number_of_words):
    """ We want to prepare the reviews to be prepared for embedding"""
    
    encoded_docs = [one_hot(d, number_of_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ') for d in documents]
    max_length = 100
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print("Reviews prepared for beeing embedded ! Shape: ")
    print(padded_docs.shape)
    return padded_docs


#########################


class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        
        # First, we will clean the training set:
        train_set = pd.read_csv(trainfile, sep='\t', header= None)
        
        # Do the cleaning: Lower cases, no stop word, no punctuation:
        aspect_term(train_set, 2)
        sentence_modifications(train_set, 4)
        
        # First, we create a tokenizer:
        voc_size = 7000
        tokenizer = Tokenizer(num_words=voc_size)
        # Give the list of texts to train our tokenizer
        tokenizer.fit_on_texts(train_set[4])
        
        # Then, we save the existing tokenizer to apply it on new data.
        with open('tokenizer.pickle', 'wb') as handle:
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
        
        
        # Time to build the model:
        
        #############################################################################
        
        #We will try to merge two different models: Accuracy: 79.79 
        left_branch = Input((7000, ))
        left_branch_dense = Dense(512, activation = 'relu')(left_branch)
        
        right_branch = Input((14012, ))
        right_branch_dense = Dense(512, activation = 'relu')(right_branch)
        merged = Concatenate()([left_branch_dense, right_branch_dense])
        output_layer = Dense(3, activation = 'softmax')(merged)
        
        model = Model(inputs=[left_branch, right_branch], outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit([np.array(review_matrix), np.array(X_train)], labels,epochs=2, verbose=1)
        model.save('model.merged') 
        
        #############################################################################
        
        
        #############################################################################

        #We will try to merge two different models in a different way: Accuracy: 70
        
        # Prepare the review column for embedding: 
        review_matrix_for_embedding = prepare_for_encoding(train_set[4].tolist(), 7000) # Shape: (1503,100)
        
        
        second_matrix = np.array(pd.concat([onehot_category, aspect_matrix],axis=1))
                
        
        left_branch = Input(shape=(100,), dtype='int32')
        # input_dim: Size of maximum integer (7001 here); output dim: Size of embedded vector; 
        # input_length: Size of the array
        left_branch_embedding = Embedding(7000, 300, input_length=100)(left_branch)
        lstm_out = LSTM(256)(left_branch_embedding)
        lstm_out = Dropout(0.7)(lstm_out)
        lstm_out = Dense(128, activation='relu')(lstm_out)
        
        right_branch = Input((7012, ))
        merged = Concatenate()([lstm_out, right_branch])
        output_layer = Dense(3, activation = 'softmax')(merged)
        
        model = Model(inputs=[left_branch, right_branch], outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit([review_matrix_for_embedding, second_matrix], labels,epochs=5, verbose=1)
        model.save('model.merged2') # 79.79 accuracy


        #############################################################################



        #############################################################################

        # Let's try a LSTM now, with different features
        # Accuracy: 70%
        #embed_dim = 128
        #lstm_out = 200

        #model = Sequential()
        #model.add(Embedding(512, embed_dim,input_length = X_train.shape[1]))
        #model.add(LSTM(lstm_out))
        #model.add(Dense(3,activation='softmax'))
        #model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        #print(model.summary())
        #model.fit(X_train, labels, epochs=2, verbose=1)
        #model.save('model.lstm')
        
        #############################################################################
        
        
        #############################################################################
        
        # First, we build the training set and the label column: Accuracy: 78.72
        
        
        model = Sequential()
        # Input of size * x 14012 and output of size 512
        model.add(Dense(512, input_shape=(14012,)))
        # Relu activation function
        model.add(Activation('relu'))
        # Final output of size 3 (three posible polarities)
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, labels, epochs=2, verbose=1)
        model.save('model.simple') # 
        
        #clf = svm.SVC(gamma='scale')
        #clf.fit(X_train, labels)
        #svm_model = pickle.dumps(clf)
        
        #############################################################################

        
        
        # At the end of the predict file, we will need to convert  int label
        # to polarities (pos, neg, neutr),thus, we save the labels:
        # Save the fitted label encoder for prediction decoding
        with open('labels.pickle', 'wb') as handle:
            	pickle.dump(polarity_encoder, handle)
            
            
        

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
        # First, we will clean the test set:
        test_set = pd.read_csv(datafile, sep='\t', header= None)

        # Do the cleaning: Lower cases, no stop word, no punctuation:
        aspect_term(test_set, 2)
        sentence_modifications(test_set, 4)
        sentence_for_embedding = prepare_for_encoding(test_set[4].tolist(), 7000) 

        # Apply tokenizer on new data
        with open('tokenizer.pickle', 'rb') as handle:
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
        
        #clf2 = pickle.loads(svm_model)
        #predictions = clf2.predict(X_train)
        merged_model = load_model('model.merged2')
        #predictions = merged_model.predict_classes([np.array(X_train.iloc[:,7012:14012]),np.array(X_train)],)
        
        # Simple sequential model without LSTM
        #predictions = merged_model.predict_classes(np.array(X_train),)
        
        
        # For functional model without LSTM
        #predictions = merged_model.predict([np.array(X_train.iloc[:,7012:14012]),np.array(X_train)])
        
        # For functional model with LSTM
        predictions = merged_model.predict([sentence_for_embedding,np.array(X_train.iloc[:,:7012])])
       
        
        predictions = predictions.argmax(axis=-1)  
        #print(predictions)
         # Inverse the labels transformation
        with open('labels.pickle', 'rb') as handle:
            polarity_labels = pickle.load(handle)
        
        
        
        return polarity_labels.inverse_transform(predictions)
        
        
        
        
        
        
# To run the code:
# On the terminal, go to the folder and then => $ python tester.py
