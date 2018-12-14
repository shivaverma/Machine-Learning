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
print(v.vocabulary_.get('midway'))       # printing index of a word
print(bag_of_word)

print('----------------------------')

text = ["hello world india"]

clf = TfidfVectorizer(stop_words="english")      # convert the words into some type of frequency
clf.fit(text)
bag = clf.transform(text)
print(clf.vocabulary_.get('world'))
print(bag)
