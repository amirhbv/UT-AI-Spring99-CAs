from random import choice, random, sample
from re import findall, split
from string import ascii_lowercase, ascii_uppercase
from typing import List
from time import time


def shuffle_str(s: str) -> str:
    return ''.join(sample(list(s), len(s)))


class Chromosome(object):

    def __init__(self, mapping: str):
        self.mapping = {
            k: v for k, v in zip(list(ascii_lowercase), list(mapping))
        }

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

    @classmethod
    def frequency_based(cls, dictionary_char_frequency, encoded_text_char_frequency):
        print(list(zip(
            sorted([(frequency, char)
                    for char, frequency in dictionary_char_frequency.items()]),
            sorted([(frequency, char)
                    for char, frequency in encoded_text_char_frequency.items()])
        )))
        mapping = {
            k: v for ((_, v), (_, k)) in zip(
                sorted([(frequency, char)
                        for char, frequency in dictionary_char_frequency.items()]),
                sorted([(frequency, char)
                        for char, frequency in encoded_text_char_frequency.items()])
            )
        }
        print(mapping)
        return cls(''.join(map(lambda c: mapping.get(c), ascii_lowercase)))

    @classmethod
    def crossover(cls, father, mother, change_threshold=0.75):
        mapping = ''

        for char in ascii_lowercase:
            father_mapping = father.mapping.get(char)
            mother_mapping = mother.mapping.get(char)

            if random() > change_threshold:
                mapping += choice(
                    ''.join(
                        filter(lambda c: c not in mapping,
                               ascii_lowercase)
                    )
                )
            elif father_mapping != mother_mapping:
                if father_mapping not in mapping:
                    mapping += father_mapping
                elif mother_mapping not in mapping:
                    mapping += mother_mapping
                else:
                    mapping += choice(
                        ''.join(
                            filter(lambda c: c not in mapping,
                                   ascii_lowercase)
                        )
                    )
            else:
                if mother_mapping not in mapping:
                    mapping += mother_mapping
                else:
                    mapping += choice(
                        ''.join(
                            filter(lambda c: c not in mapping,
                                   ascii_lowercase)
                        )
                    )

            # if father_mapping != mother_mapping:
            #     if father_mapping not in mapping:
            #         mapping += father_mapping
            #     elif mother_mapping not in mapping:
            #         mapping += mother_mapping
            #     else:
            #         mapping += choice(
            #             ''.join(
            #                 filter(lambda c: c not in mapping,
            #                        ascii_lowercase)
            #             )
            #         )
            # else:
            #     if random() < change_threshold and mother_mapping not in mapping:
            #         mapping += mother_mapping
            #     else:
            #         mapping += choice(
            #             ''.join(
            #                 filter(lambda c: c not in mapping,
            #                        ascii_lowercase)
            #             )
            #         )

        return cls(mapping)

    def __str__(self):
        return ''.join(self.mapping.values())


class Decoder(object):

    def __init__(self, encoded_text, population_size: int = 150):
        self.encoded_text: str = encoded_text

        self.dictionary = {
            k: True for k in filter(
                lambda word: len(word) > 1,
                split('\W+', open('global_text.txt').read().lower())
            )
        }

        encoded_text_char_frequency = Decoder.count_char_frequency(
            ''.join(
                filter(
                    lambda word: len(word) > 1,
                    split('\W+', self.encoded_text.lower())
                )
            )
        )
        dictionary_char_frequency = Decoder.count_char_frequency(
            ''.join(self.dictionary.keys())
        )

        print(encoded_text_char_frequency)
        print(dictionary_char_frequency)

        self.population: List[Chromosome] = [
            Chromosome.frequency_based(
                dictionary_char_frequency,
                encoded_text_char_frequency
            ) if random() > 0.3 else Chromosome.random() for i in range(population_size)
        ]
        self.population_size: int = population_size
        self.fitness_cache = {}

    @classmethod
    def count_char_frequency(cls, text):
        char_count = {
            c: 0 for c in ascii_lowercase
        }

        for c in text:
            if c.isalpha():
                char_count[c.lower()] += 1

        return char_count

    def calculate_fitness(self, chromosome: Chromosome):
        mapping = str(chromosome)
        if mapping in self.fitness_cache:
            return self.fitness_cache.get(mapping)

        decoded_text = chromosome.decode(self.encoded_text)
        decoded_words = findall('\w+', decoded_text)
        wrong_words = set([
            word for word in decoded_words if len(word) > 1 and word.lower() not in self.dictionary
        ])
        fitness = len(wrong_words)
        if fitness < 20:
            print(wrong_words)
        self.fitness_cache[mapping] = fitness
        return fitness

    def decode(self) -> str:
        found = False
        generation = 0
        start = time()

        repeat_count = 0
        best_fitness = 9999

        while True:
            population = sorted(self.population, key=self.calculate_fitness)
            f = self.calculate_fitness(population[0])
            if f < best_fitness:
                best_fitness = f
                repeat_count = 0
            elif f == best_fitness:
                repeat_count += 1

            print(generation, f, population[0])
            if f < 2:
                print(population[0])
                print(time() - start)
                return population[0].decode(self.encoded_text)

            self.population = []
            for i, chromosome in enumerate(population):
                if i < (self.population_size / 5):
                    self.population.append(chromosome)
                else:
                    father, mother = sample(population[:60], 2)
                    if repeat_count > 100:
                        change_threshold = 0.00
                    elif repeat_count > 70:
                        change_threshold = 0.01
                    elif repeat_count > 60:
                        change_threshold = 0.05
                    elif repeat_count > 50:
                        change_threshold = 0.15
                    elif repeat_count > 40:
                        change_threshold = 0.25
                    elif repeat_count > 30:
                        change_threshold = 0.45
                    elif repeat_count > 20:
                        change_threshold = 0.55
                    elif repeat_count > 10:
                        change_threshold = 0.65
                    else:
                        change_threshold = 0.75

                    self.population.append(
                        Chromosome.crossover(father, mother, change_threshold)
                    )

            generation += 1
