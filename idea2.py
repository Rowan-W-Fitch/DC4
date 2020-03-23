import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve

csv = sys.argv[1]
#create panada df
df = pd.read_csv(csv, encoding = 'utf-8', delimiter = "\t", header = 0)
#removing unnceccessary columns and null values
df.dropna(axis = 0, inplace = True)
df.drop(['id', 'qid1', 'qid2'], axis = 1, inplace = True)
print('df read') #status update
#clean pd data
def clean_data(question):
    #handle null value
    if pd.isnull(question):
        return ""
    #handle empty question
    if type(question) != str or question == "":
        return ""
    """ large list of cleaning the question, confusing ones are explained"""
    #handle abbreviations
    question = re.sub("\'ve", " have ", question)
    question = re.sub("can't", "can not", question)
    question = re.sub("n't", " not ", question)
    question = re.sub("i'm", "i am", question, flags=re.IGNORECASE)
    question = re.sub("\'re", " are ", question)
    question = re.sub("\'d", " would ", question)
    question = re.sub("\'ll", " will ", question)
    question = re.sub("e\.g\.", " eg ", question, flags=re.IGNORECASE)
    question = re.sub("b\.g\.", " bg ", question, flags=re.IGNORECASE)
    #handle referring to thousand as K (ie 50K -> 50000)
    question = re.sub("(\d+)(kK)", " \g<1>000 ", question)
    #standardize email
    question = re.sub("e-mail", " email ", question, flags=re.IGNORECASE)
    #handle differnet ways of referring to the USA
    question = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", question, flags=re.IGNORECASE)
    question = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", question, flags=re.IGNORECASE)
    question = re.sub("\(s\)", " ", question, flags=re.IGNORECASE)
    question = re.sub("[c-fC-F]\:\/", " disk ", question)
    #remove commas btwn numbers
    question = re.sub('(?<=[0-9])\,(?=[0-9])', "", question)
    #common chars subbed for word
    question = re.sub('\$', " dollar ", question)
    question = re.sub('\%', " percent ", question)
    question = re.sub('\&', " and ", question)
    #handling Indian dollar
    question = re.sub("(?<=[0-9])rs ", " rs ", question, flags=re.IGNORECASE)
    question = re.sub(" rs(?=[0-9])", " rs ", question, flags=re.IGNORECASE)
    #other subs from https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    #proper capitalization of countries
    question = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", question)
    question = re.sub(r" UK ", " England ", question, flags=re.IGNORECASE)
    question = re.sub(r" india ", " India ", question)
    question = re.sub(r" switzerland ", " Switzerland ", question)
    question = re.sub(r" china ", " China ", question)
    question = re.sub(r" chinese ", " Chinese ", question)
    #handle capitalization and spelling
    question = re.sub(r" imrovement ", " improvement ", question, flags=re.IGNORECASE)
    question = re.sub(r" intially ", " initially ", question, flags=re.IGNORECASE)
    question = re.sub(r" quora ", " Quora ", question, flags=re.IGNORECASE)
    question = re.sub(r" dms ", " direct messages ", question, flags=re.IGNORECASE)
    question = re.sub(r" demonitization ", " demonetization ", question, flags=re.IGNORECASE)
    question = re.sub(r" actived ", " active ", question, flags=re.IGNORECASE)
    question = re.sub(r" kms ", " kilometers ", question, flags=re.IGNORECASE)
    question = re.sub(r" cs ", " computer science ", question, flags=re.IGNORECASE)
    question = re.sub(r" upvote", " up vote", question, flags=re.IGNORECASE)
    question = re.sub(r" iPhone ", " phone ", question, flags=re.IGNORECASE)
    question = re.sub(r" \0rs ", " rs ", question, flags=re.IGNORECASE)
    question = re.sub(r" calender ", " calendar ", question, flags=re.IGNORECASE)
    question = re.sub(r" ios ", " operating system ", question, flags=re.IGNORECASE)
    question = re.sub(r" gps ", " GPS ", question, flags=re.IGNORECASE)
    question = re.sub(r" gst ", " GST ", question, flags=re.IGNORECASE)
    question = re.sub(r" programing ", " programming ", question, flags=re.IGNORECASE)
    question = re.sub(r" bestfriend ", " best friend ", question, flags=re.IGNORECASE)
    question = re.sub(r" dna ", " DNA ", question, flags=re.IGNORECASE)
    question = re.sub(r" III ", " 3 ", question)
    question = re.sub(r" banglore ", " Banglore ", question, flags=re.IGNORECASE)
    question = re.sub(r" J K ", " JK ", question, flags=re.IGNORECASE)
    question = re.sub(r" J\.K\. ", " JK ", question, flags=re.IGNORECASE)

    question = ''.join([c for c in question if c not in punctuation]).lower()
    return question

#clean the pd
df['question1'] = df['question1'].apply(clean_data)
df['question2'] = df['question2'].apply(clean_data)
print('data cleaned') #status update

#getting TFIDF
tfidf = TfidfVectorizer(analyzer = 'word', max_features = 5000)
tfidf.fit(pd.concat((df['question1'],df['question2'])).unique())
TFIDF_Q1 = tfidf.transform(df['question1'].values)
TFIDF_Q2 = tfidf.transform(df['question2'].values)
#setting up data for gradient boost
X = scipy.sparse.hstack((TFIDF_Q1, TFIDF_Q2))
Y = df['is_duplicate'].values
X_tr, X_tst, Y_tr, Y_tst = train_test_split(X, Y, test_size = 0.3, random_state = 47)
#create gradient boost model instance
print('training model') #status update
gb = xgb.XGBClassifier(
                        max_depth=50,
                        n_estimators=80,
                        learning_rate=0.1,
                        colsample_bytree=.7,
                        gamma=0, reg_alpha=4,
                        objective='binary:logistic',
                        eta=0.3, silent=1,
                        subsample=0.8
                        ).fit(X_tr, Y_tr)

y_out = gb.predict(X_tst)
#get metrics
recall_scr = recall_score(Y_tst, y_out, average = 'micro' )
f1_scr = f1_score(Y_tst, y_out, average = 'micro')
pr_score = precision_score(Y_tst, y_out, average = 'micro')
print("recall: " + str(recall_scr))
print("f1: " + str(f1_scr))
print("precision: " + str(pr_score))
#plot
precision, recall, r = precision_recall_curve(Y_tst, y_out)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.show()

#put two questions, see if they're duplicate
question_1 = input('enter a question: ')
question_2 = input('enter another question: ')
q1 = clean_data(question_1)
q2 = clean_data(question_2)
d = {'q1': [q1], 'q2':[q2]}
df2 = pd.DataFrame(data = d)
tf1 = tfidf.transform(df2['q1'].values)
tf2 = tfidf.transform(df2['q2'].values)
input = scipy.sparse.hstack((tf1,tf2))
out = gb.predict(input)
print()
if out[0] == 0:
    print('different')
else:
    print("marked as duplicate")
