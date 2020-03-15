import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO

import nltk
nltk.download('punkt')


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        # caption = str(coco.anns[id]['caption'])
        caption = str(coco.anns[id]['tokenized_caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def build_vocab_custom(tokenized_text_list, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for i, tokens in enumerate(tokenized_text_list):
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(tokenized_text_list)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p 


def main(args):
    # vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    tokenized_text_list = unpickle('tokenized_bokete_text.pkl')
    vocab = build_vocab_custom(tokenized_text_list, 2)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    """

    # for japanese
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/stair_captions_v1.2_train_tokenized.json',
                        help='path for train annotation file')
    # parser.add_argument('--vocab_path', type=str, default='./data/vocab_jp.pkl',
    #                     help='path for saving vocabulary wrapper')

    parser.add_argument('--vocab_path', type=str, default='./data/vocab_jp_bokete.pkl',
                        help='path for saving vocabulary wrapper')

    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)

    '''
    json=args.caption_path
    threshold = 4

    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        print(str(coco.anns[id]['caption']))
        caption = str(coco.anns[id]['tokenized_caption'])
        print(caption)
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        print(tokens)
        counter.update(tokens)
        if i == 50:
            break

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    print(words)
    print(len(words))
    '''


