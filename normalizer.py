import pymorphy2
import string


class Normalizer(object):

    def __init__(self, lang='ru', space_chars='\t\n\r', strip_chars=string.punctuation, min_word_length=4):
        if lang != 'ru':
            raise NotImplementedError('See https://github.com/kmike/pymorphy2/issues/80#issuecomment-269759417')
        
        self.min_word_length = min_word_length
        self.morphper = pymorphy2.MorphAnalyzer()
        self.translator = str.maketrans(space_chars, ' '*len(space_chars), strip_chars)
        
    def normalize(self, sentense):
        sentense = sentense.translate(self.translator)  # remove punctuation, replace tabs & newlines with space
        sentense = ' '.join(sentense.split())  # remove extra spaces
        words = sentense.split(' ')
        norm_words = [
            self.morphper.parse(word)[0].normal_form for word in words 
            if len(word) >= self.min_word_length
        ]
        return ' '.join(norm_words).strip().lower()
