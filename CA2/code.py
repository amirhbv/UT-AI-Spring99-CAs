from re import match, split
from typing import List


class Decoder(object):

    def __init__(self, encoded_text):
        self.encoded_text: str = encoded_text

        stop_words = Decoder.generate_stop_words()
        global_text = split('\W', open('global_text.txt').read())

        self.dictionary: List[str] = list(
            filter(
                lambda word: len(word) > 1 and word not in stop_words,
                global_text
            )
        )

    def decode(self) -> str:
        return ''

    @classmethod
    def generate_stop_words(cls) -> List[str]:
        stop_words: List[str] = []
        try:
            with open('stop_words.txt') as fin:
                for line in fin:
                    m = match('(^\w+)', line)
                    if m:
                        stop_words.append(m.group(1))

        except Exception as e:
            pass

        return stop_words
