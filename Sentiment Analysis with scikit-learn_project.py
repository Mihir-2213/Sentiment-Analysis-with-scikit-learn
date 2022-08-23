## Loading the dataset

import pandas as pd

df = pd.read_csv('movie_data.csv')
df.head(10)

df['review'][0] #the last sentence says my vote is 7 which makes sense that classification above shows positive

##Transforming documents into feature vectors
#Lyrics of bob marley song
#we want to change into sparse vector by using bag of words method
#input this document, transform it into numerical values

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array(['The sun is shining',
                'The weather is sweet',
                'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)   #vocabulary is saved in python dictionary
                                  #which maps the unique words in the docs to integer indeces
print(count.vocabulary_)
print(bag.toarray())

#N.B. Term frequency is number a word occurs in a document (count vector)
# 'is' term frequency in 3rd sentence is 3
#TFidf = Term Frequency Inverse Document Frequency
# It is technique used to downweight those frequently occurring words in the feature vectors.

## Word relevancy using term frequency-inverse document frequency
# tf-idf is used to downweight the most frequent occuring words
#nd is = 3 here
# denominator is 1 to ensure non zeor
#logarithm is used to ensure words are not given too much weights

from sklearn.feature_extraction.text import TfidfTransformer

np.set_printoptions(precision=2)   #so array looks non-messy

tfidf = TfidfTransformer(use_idf = True, #enable reweighing
                        norm = 'l2',   #each output will have a unit norm, here sum of squares of vector elements = 1
                        smooth_idf = True   #weights by adding 1 to the document frequencies as if seen once more so avoid error of division by 0
                        )

print(tfidf.fit_transform(bag).toarray()) # 'is' term frequency in 3rd sentence is deflated to 0.45 because repeated in sentence 1&2

## Data Preparation

df.loc[0, 'review'][-50:] #the last 50 characters
                           #i have html, tags, commas and emojis might also be available

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) #replace html tags with a empty string
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) #get text emojes like smiley face, disappointed, sad
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '') #we will move the emojis to end of text review
    return text

preprocessor(df.loc[0, 'review'][-50:])
#we will notice now panctuation like full stop, colon is not included
#also html tags and whats in between them have also been removed

preprocessor("<,/a> this :) is a :( test :-) !")

df['review'] = df['review'].apply(preprocessor)

## Tokenization of documents

#stemming to remove derivations of common words to a base word like organize, organized, organizer, organization
# stemming chops words and remove derivations
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

#tokenize the text and split the sentence into words according to occurance
#stemming technique

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return[porter.stem(word) for word in text.split()]

tokenizer('runners like running and thus they run')

tokenizer_porter('runners like running and thus they run')

#the ends of words have been stripped
#idiosyncrncy like thus is changed to thu

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and thus she always runs a lot')[-10:] if w not in stop]

#removed 'a', 'and'

## Transform Text Data into TF-IDF Vectors

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents = None,
                        lowercase = False,
                        preprocessor = None, #because we already did on our data
                        tokenizer = tokenizer_porter, #stemming function
                        use_idf = True,    #to downgrade according to frequency
                        norm = 'l2',
                        smooth_idf = True   #to avoid division by zero
                       )

#split to x, y components
y = df.sentiment.values   #numpy array
x = tfidf.fit_transform(df.review)

## Document Classification using Logistic Regression

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y,
                                                    random_state = 1, #to get same as instructor
                                                    test_size = 0.5,
                                                    shuffle = False
                                                   )

import pickle
#to dump our model on the disk
from sklearn.linear_model import LogisticRegressionCV
#Log Regression has hyperparemters and instead of manually trying to fine tune it
#we can use estimator of sklearn CV
#The output of a logistic model can be interpreted as a probability.

#Cross validation can be used to:
#1- Tune Model Hyperparamters
#2- Assess model performance out of sample

clf = LogisticRegressionCV(cv = 5, #cross validation fold
                          scoring = 'accuracy',
                          random_state = 0,
                          n_jobs = -1, #to deticate all our CPU to solving this task
                          verbose = 3, #to see output while doing computations
                          max_iter = 300    #for safety of convergence
                          ).fit(X_train, Y_train)

saved_model = open('saved_model.sav', 'wb')  #write bytes to this file
pickle.dump(clf, saved_model)
saved_model.close()

## Model Evaluation

filename = 'saved_model.sav'
saved_clf = pickle.load(open(filename, 'rb'))   #load from disk the saved model

saved_clf.score(X_test, Y_test)






