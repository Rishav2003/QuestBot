import re
from bs4 import BeautifulSoup
def preprocess(q):
    
    q = str(q).lower().strip()
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    #Both good sources 
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    #Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    #Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q
import pandas as pd 
def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)  

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1) + len(w2)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']

    SAFE_DIV = 0.0001

    STOP_WORDS = stopwords.words("english")

    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

import distance

def fetch_length_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    #Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    #Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    
    return length_features
    
# Fuzzy Features
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    fuzzy_features = [0.0]*4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    #This is  fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    #This is  token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    #This is token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import pickle
def get_vector(sentence):
    vec_sum = np.zeros(model.vector_size)
    count = 0
    for word in sentence:
        if word in model.wv:
            vec_sum += model.wv[word]
            count += 1
    return vec_sum / count if count != 0 else np.zeros(model.vector_size)

def generate_feature(q1,q2,model):
    q1=preprocess(q1)
    q2=preprocess(q2)
    
    balanced_df=pd.DataFrame()
    data = {'question1': [q1], 'question2': [q2]}
    balanced_df = pd.DataFrame(data)
    balanced_df['q1_len'] = balanced_df['question1'].str.len()
    balanced_df['q2_len'] = balanced_df['question2'].str.len()
    balanced_df['q1_num_words'] = balanced_df['question1'].apply(lambda row: len(row.split(" ")))
    balanced_df['q2_num_words'] = balanced_df['question2'].apply(lambda row: len(row.split(" ")))
    balanced_df['word_common'] = balanced_df.apply(common_words, axis=1)
        
    balanced_df['word_total'] = balanced_df.apply(total_words, axis=1)

    balanced_df['word_share'] = round(balanced_df['word_common'] / balanced_df['word_total'], 2)

    token_features = balanced_df.apply(fetch_token_features, axis=1)

    balanced_df["cwc_min"] = list(map(lambda x: x[0], token_features))
    balanced_df["cwc_max"] = list(map(lambda x: x[1], token_features))
    balanced_df["csc_min"] = list(map(lambda x: x[2], token_features))
    balanced_df["csc_max"] = list(map(lambda x: x[3], token_features))
    balanced_df["ctc_min"] = list(map(lambda x: x[4], token_features))
    balanced_df["ctc_max"] = list(map(lambda x: x[5], token_features))
    balanced_df["last_word_eq"] = list(map(lambda x: x[6], token_features))
    balanced_df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    length_features = balanced_df.apply(fetch_length_features, axis=1)
    balanced_df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
    balanced_df['mean_len'] = list(map(lambda x: x[1], length_features))
    balanced_df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))
    
    fuzzy_features = balanced_df.apply(fetch_fuzzy_features, axis=1)
    balanced_df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    balanced_df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    balanced_df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    balanced_df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))
    q_df = balanced_df[['question1','question2']]
    final_df = balanced_df.drop(columns=['question1','question2'])
    sentences = [str(sentence).split() for sentence in q_df['question1']] + [str(sentence).split() for sentence in q_df['question2']]
    q1_arr= q_df['question1'].apply(lambda x: get_vector(str(x).split()))
    q2_arr = q_df['question2'].apply(lambda x: get_vector(str(x).split()))
    q1_arr=list(q1_arr)
    columns = [f'feature_{i+1}' for i in range(len(q1_arr[0]))]
    temp_df1 = pd.DataFrame(q1_arr, columns=columns)
    q2_arr=list(q2_arr)
    columns2 = [f'feature_{i+1}_2' for i in range(len(q2_arr[0]))]
    temp_df2 = pd.DataFrame(q2_arr, columns=columns2)
    temp_df = pd.concat([temp_df1, temp_df2], axis=1)
    final_df = pd.concat([final_df, temp_df], axis=1)
    return final_df



ques1 = "My name is Rishav"
ques2 = "What are the steps to build a machine learning model?"
model = pickle.load(open('word2vec_model.pkl','rb'))
x=generate_feature(ques1,ques2,model)
print(type(x))
print(x)