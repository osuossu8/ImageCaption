import html
import mojimoji
import re
import MeCab
import numpy as np


class MecabTokenizer(object):
    def __init__(self):
        self.wakati = MeCab.Tagger('-Owakati')
        self.wakati.parse('')

    def tokenize(self, line):
        return self.wakati.parse(line).strip().split(" ")


class TextPreprocessorJP(object):
    def __init__(self):
        self.puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
                       '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',
                       '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
                       '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
                       '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '（', '）', '～',
                       '➡', '％', '⇒', '▶', '「', '➄', '➆',  '➊', '➋', '➌', '➍', '⓪', '①', '②', '③', '④', '⑤', '⑰', '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽',  
                       '＝', '※', '㈱', '､', '△', '℮', 'ⅼ', '‐', '｣', '┝', '↳', '◉', '／', '＋', '○',
                       '【', '】', '✅', '☑', '➤', 'ﾞ', '↳', '〶', '☛', '｢', '⁺', '『', '≫',
                       ] 

        self.html_tags = ['<p>', '</p>', '<table>', '</table>', '<tr>', '</tr>', '<ul>', '<ol>', '<dl>', '</ul>', '</ol>',
                          '</dl>', '<li>', '<dd>', '<dt>', '</li>', '</dd>', '</dt>', '<h1>', '</h1>',
                          '<br>', '<br/>', '<strong>', '</strong>', '<span>', '</span>', '<blockquote>', '</blockquote>',
                          '<pre>', '</pre>', '<div>', '</div>', '<h2>', '</h2>', '<h3>', '</h3>', '<h4>', '</h4>', '<h5>', '</h5>',
                          '<h6>', '</h6>', '<blck>', '<pr>', '<code>', '<th>', '</th>', '<td>', '</td>', '<em>', '</em>']

        self.spaces = ['\u200b', '\u200e', '\u202a', '\u2009', '\u2028', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u3000', '\x10', '\x7f', '\x9d', '\xad',
                       '\x97', '\x9c', '\x8b', '\x81', '\x80', '\x8c', '\x85', '\x92', '\x88', '\x8d', '\x80', '\x8e', '\x9a', '\x94', '\xa0', 
                       '\x8f', '\x82', '\x8a', '\x93', '\x90', '\x83', '\x96', '\x9b', '\x9e', '\x99', '\x87', '\x84', '\x9f',
                      ]

        self.numbers = ["0","1","2","3","4","5","6","7","8","9","０","１","２","３","４","５","６","７","８","９"]

    def _pre_preprocess(self, x):
        return str(x).lower() 

    def rm_spaces(self, x):
        """
        空白スペースを除去する.
        """
        for space in self.spaces:
                x = x.replace(space, ' ')
        return x

    def clean_html_tags(self, x, stop_words=[]):  
        """
        html tag を削除する.
        """    
        for r in self.html_tags:
            x = x.replace(r, '')
        for r in stop_words:
            x = x.replace(r, '')
        return x

    def replace_num(self, x):
        """
        数値は除去する.
        """
        x = re.sub('[0-9]{5,}', '', x)
        x = re.sub('[0-9]{4}', '', x)
        x = re.sub('[0-9]{3}', '', x)
        x = re.sub('[0-9]{2}', '', x)
        return x

    def clean_puncts(self, x):
        """
        句読点の間隔を空ける.
        """
        for punct in self.puncts:
            x = x.replace(punct, f' {punct} ')
        return x

    def clean_text_jp(self, x):
        x = x.replace('。', '')
        x = x.replace('、', '')
        x = x.replace('\n', '') # 改行削除
        x = x.replace('\t', '') # タブ削除
        x = x.replace('\r', '')
        x = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', x) 
        x = re.sub(r'\[math\]', ' LaTex math ', x) # LaTex削除
        x = re.sub(r'\[\/math\]', ' LaTex math ', x) # LaTex削除
        x = re.sub(r'\\', ' LaTex ', x) # LaTex削除
        x = re.sub(' +', ' ', x)
        return x   

    def han_to_zen(self, x):
        word = []
        for x_s in x.split():
            x_s = mojimoji.han_to_zen(x_s)
            word.append(x_s)  
        return " ".join(word)

    def preprocess(self, sentence):
        sentence = sentence.fillna(" ")
        sentence = sentence.apply(lambda x: self._pre_preprocess(x))
        sentence = sentence.apply(html.unescape) # html エスケープ文字列を元に戻す
        sentence = sentence.apply(lambda x: self.rm_spaces(x))
        sentence = sentence.apply(lambda x: self.clean_puncts(x))
        #sentence = sentence.apply(lambda x: self.replace_num(x))
        sentence = sentence.apply(lambda x: self.clean_html_tags(x))
        sentence = sentence.apply(lambda x: self.clean_text_jp(x))
        #sentence = sentence.apply(lambda x: self.han_to_zen(x))
        return sentence
