# author: Shiva Verma, Mail: shivajbd@gmail.com
# bag of words, count vectorizer

from nltk.stem.snowball import SnowballStemmer

string1 = "in the midway of a jungle she saw front she a were"
string2 = "for the shake of ram of in ram with you"

sm = SnowballStemmer("english")
print(sm.stem(string1))

