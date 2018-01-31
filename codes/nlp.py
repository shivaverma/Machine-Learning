from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def tokenize(data):

    x = word_tokenize(data)
    return x


def stop_words(data):

    w = set(stopwords.words('english'))
    extra = {' ', '.', ',', ';', ':', '-'}
    w = w.union(extra)
    x = [i for i in data if i not in w]
    return x


def stem(data):

    ps = PorterStemmer()
    x = [str(ps.stem(i)) for i in data]
    return x


if __name__ == '__main__':

    f = open('random_story.txt', 'r')
    txt = f.read()
    token = tokenize(txt)
    token = stop_words(token)
    token = stem(token)

