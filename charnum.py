import numpy as np

class CharNumberEncoder(object):

    def __init__(self, data_iterator, char_map=None, word_len=30, sent_len=200):
        '''
        DESCRIPTIONS:
            This class converts text to numbers for the standard unicode vocabulary
            size.
        PARAMS:
            data_iterator (iterator): iterator to iterates the text strings
            word_len (int): maximum length of the word, any word of length less
                than that will be padded with zeros, any word of length more than
                that will be cut at max word length.
            sent_len (int): maximum number of words in a sentence, any sentence
                with less number of words than that will be padded with zeros,
                any sentence with more words than the max number will be cut at
                the max sentence length.
            char_map (dict): a dictionary for mapping characters to numbers.
        '''
        self.data_iterator = data_iterator
        self.word_len = word_len
        self.sent_len = sent_len
        self.char_map = char_map


    def build_char_map(self):
        char_set = set()
        for paragraph in self.data_iterator:
            for c in str(paragraph):
                char_set.add(c)
        self.char_map = {}
        i = 1
        for c in char_set:
            if c == ' ':
                self.char_map[c] = 0
            else:
                self.char_map[c] = i
                i += 1
        return self.char_map


    def make_char_embed(self):
        '''build array vectors of words and sentence, automatically skip non-ascii
           words.
        '''
        if self.char_map is None:
            print '..no char_map, building new character map'
            self.build_char_map()

        print '..total {} characters in char_map'.format(len(self.char_map))

        sents = []
        char_set = set()
        for paragraph in self.data_iterator:
            word_toks = str(paragraph).split(' ')
            word_vecs = []
            for word in word_toks:
                word = word.strip()
                word_vec = []
                for c in word:
                    if c not in self.char_map:
                        print '{} not in character map'.format(c)
                    else:
                        word_vec.append(self.char_map[c])

                if len(word_vec) > 0:
                    word_vecs.append(self.spawn_word_vec(word_vec))
            if len(word_vecs) > self.sent_len:
                sents.append(word_vecs[:self.sent_len])
            else:
                zero_pad = np.zeros((self.sent_len-len(word_vecs), self.word_len))
                if len(word_vecs) > 0:
                    sents.append(np.vstack([np.asarray(word_vecs), zero_pad]))
                else:
                    sents.append(zero_pad)
        return np.asarray(sents)


    def spawn_word_vec(self, word_vec):
        '''Convert a word to number vector with max word length, skip non-ascii
           characters
        '''
        if len(word_vec) > self.word_len:
            return word_vec[:self.word_len]
        else:
            word_vec += [0]*(self.word_len-len(word_vec))
        return word_vec
