# author: Shiva Verma, Mail: shivajbd@gmail.com
# bag of words, count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer()                    # counts words in documents

string1 = "in the midway of a jungle she saw front she a was"
string2 = "for the shake of ram of in ram with you"

text_list = [string1, string2]
v.fit(text_list)                         # fitting the document
bag_of_word = v.transform(text_list)     # transforming to count
print bag_of_word
