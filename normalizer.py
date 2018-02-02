import pymorphy2
import string
import re


class Normalizer(object):

    def __init__(self, lang='ru', space_chars='\t\n\r', strip_chars=string.punctuation+'«»', min_word_length=4, max_word_length=15):
        if lang != 'ru':
            raise NotImplementedError('See https://github.com/kmike/pymorphy2/issues/80#issuecomment-269759417')
        
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.morphper = pymorphy2.MorphAnalyzer()
        self.translator = str.maketrans(space_chars, ' '*len(space_chars), strip_chars)
        self.stop_word_whitelist = ['не']  # don't remove this words - с "не" хуже определяет "locked" 
        # добавить ещё "нет"? 
        # "Спасибо,претензий к товару нет... Правда фирма немецкая,а товар сделан в Китае." ->  товар правда фирма немецкая товар сделать китай
        # + отрезало по длине "Спасибо,претензий" т.к. вырезало запятую, а пробела вокруг нет

        self.re_alphanum_only = re.compile("^[A-Za-z0-9_№\-]+$", flags=re.UNICODE)
    
    def good_word(self, word):
        if word in self.stop_word_whitelist:
            return True
        elif self.re_alphanum_only.match(word):
            return False
        elif self.min_word_length <= len(word) <= self.max_word_length:
            return True

    # Может вернуть пустую строку    
    def normalize(self, sentense):
        sentense = sentense.translate(self.translator)  # remove punctuation, replace tabs & newlines with space
        sentense = ' '.join(sentense.split())  # remove extra spaces
        words = sentense.split(' ')
        norm_words = [
            self.morphper.parse(word)[0].normal_form for word in words if self.good_word(word)
        ]
        return ' '.join(norm_words).strip('\n\r ').lower()
