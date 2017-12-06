# author: Shiva Verma, Mail: shivajbd@gmail.com
# bag of words, count vectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

v = CountVectorizer()                    # counts words in documents

string1 = "midway in in hello zebra hello for which you know you"
string2 = "for the mindset of the in the bag bag"

text_list = [string1, string2]

v.fit(text_list)                         # fitting the document
bag_of_word = v.transform(text_list)     # transforming to count object
print v.vocabulary_.get('zebra')         # printing index of a word
print bag_of_word

print '----------------------------'

clf = TfidfVectorizer(stop_words="english")
clf.fit(text_list)
bag = clf.transform(text_list)
print clf.vocabulary_.get('the')
print bag
