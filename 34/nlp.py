"""
Natural Language Processing
"""
# Importing the libraries
from re import sub
from pandas import read_csv
from numpy import concatenate
from nltk import download
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Cleaning the text
download("stopwords")
corpus = []
for i in range(len(dataset)):
    REVIEW = sub("[^a-zA-Z]", " ", dataset["Review"][i])
    REVIEW = REVIEW.lower()
    REVIEW = REVIEW.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    REVIEW = [ps.stem(word) for word in REVIEW if not word in set(all_stopwords)]
    REVIEW = " ".join(REVIEW)
    corpus.append(REVIEW)

# Cleaning the Bags of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
con = concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(con[:99])

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)
