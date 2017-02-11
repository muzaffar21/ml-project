from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from sklearn.naive_bayes import MultinomialNB

corpus = []
label_array = []

with open("/disk2/home/muzaffar/PycharmProjects/ski_learn/trainingdata.list", 'r') as tf:
    lines = tf.readlines()
label_map = {}
label_number = -1
for line in lines:
    arr = line.split("##")
    label = arr[0].strip().lower()
    document = arr[1].strip()
    # print label
    if label not in label_map:
        label_number += 1
        label_map[label] = label_number
        label_array.append(label_number)
    else:
        label_array.append(label_map[label])

    corpus.append(document)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english',
                      lowercase=True, token_pattern='[a-zA-Z0-9]+', strip_accents='unicode',
                      tokenizer=word_tokenize)

train_matrix = vec.fit_transform(corpus)
print label_map
clf = MultinomialNB()
clf.fit(train_matrix, label_array)
pickle.dump(vec, open("/disk2/home/muzaffar/PycharmProjects/ski_learn/vectorizer.pickle", "wb"))
test = ['this is really shit and bad and ']
k = vec.transform(test)
print "before.........."
print  k
l = clf.predict(k)
print l

pickle.dump(clf, open("/disk2/home/muzaffar/PycharmProjects/ski_learn/model.txt", 'wb'))

vector = pickle.load(open("/disk2/home/muzaffar/PycharmProjects/ski_learn/vectorizer.pickle", "rb"))
k2 = vector.transform(test)
print "after.........."
print  k2

model = pickle.load(open("/disk2/home/muzaffar/PycharmProjects/ski_learn/model.txt", 'rb'))
l2 = model.predict(k2)
print l2
