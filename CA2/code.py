from random import sample
from re import findall, split
from string import ascii_lowercase, ascii_uppercase
from typing import List


def shuffle_str(s: str) -> str:
    return ''.join(sample(list(s), len(s)))

class Chromosome(object):

    def __init__(self, mapping: str):
        self.mapping = {
            k: v for k, v in zip(list(ascii_lowercase), list(mapping))
        }
        pass

    def decode_char(self, encoded_char):
        if encoded_char.isalpha():
            if encoded_char.isupper():
                return self.mapping.get(encoded_char.lower()).upper()
            else:
                return self.mapping.get(encoded_char)
        else:
            return encoded_char

    def decode(self, encoded_text):
        return ''.join([self.decode_char(char) for char in encoded_text])

    @classmethod
    def random(cls):
        return cls(shuffle_str(ascii_lowercase))


class Decoder(object):

    def __init__(self, encoded_text):
        self.encoded_text: str = encoded_text

        self.dictionary: List[str] = list(
            set(
                filter(
                    lambda word: len(word) > 1,
                    split('\W+', open('global_text.txt').read().lower())
                )
            )
        )

    def calculate_fitness(self, chromosome: Chromosome):
        decoded_text = chromosome.decode(self.encoded_text)
        decoded_words = findall('\w+', decoded_text)
        return len(set(
            filter(
                lambda word: len(word) > 1 and word.lower() not in self.dictionary,
                decoded_words
            )
        ))

    def decode(self) -> str:
        print(self.calculate_fitness(Chromosome.random()))
        print(self.calculate_fitness(Chromosome('orsfwmbtizghknvelpdjcuyqax')))

        # return Chromosome('orsfwmbtizghknvelpdjcuyqax').decode(self.encoded_text)
