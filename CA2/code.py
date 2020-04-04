from collections import Counter
from math import log2
from random import choice, random, sample
from re import findall, split
from string import ascii_lowercase, ascii_uppercase
from time import time
from typing import List

DEFAULT_POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.5
MAX_REPEAT_COUNT = 100


def shuffle_str(s: str) -> str:
    return ''.join(sample(list(s), len(s)))


def pairwise(s: str, n):
    n -= 1
    prev = s[:n]

    for item in s[n:]:
        yield prev + item
        prev = prev[1:] + item


def ngram(text: str, n: int = 2):
    counter = Counter()
    words = split('\W+', text.lower())

    for word in words:
        for pair in pairwise(word, n):
            counter[pair] += 1

    return counter


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
    def crossover(cls, father, mother):
        def generate_mapping(father, mother, first_point, second_point, max_len):
            mapping = [' '] * max_len
            for i in range(first_point, second_point):
                mapping[i] = father.mapping.get(ascii_lowercase[i])

            i = j = second_point
            while j != first_point:
                mother_mapping = mother.mapping.get(ascii_lowercase[i])
                if mother_mapping not in mapping:
                    mapping[j] = mother_mapping
                    j = (j + 1) % max_len

                i = (i + 1) % max_len

            if random() < MUTATION_PROBABILITY:
                x, y = sorted(
                    sample(range(max_len), 2)
                )
                mapping[x], mapping[y] = mapping[y], mapping[x]

            return ''.join(mapping)

        if random() < CROSSOVER_PROBABILITY:
            max_len = len(ascii_lowercase)
            first_point, second_point = sorted(
                sample(range(max_len), 2)
            )

            boy_mapping = generate_mapping(
                father, mother, first_point, second_point, max_len
            )
            girl_mapping = generate_mapping(
                mother, father, first_point, second_point, max_len
            )

            return cls(boy_mapping), cls(girl_mapping)
        else:
            return father, mother

    def __str__(self):
        return ''.join(self.mapping.values())


class Decoder(object):

    def __init__(self, encoded_text, population_size: int = DEFAULT_POPULATION_SIZE):
        self.encoded_text: str = encoded_text
        self.ref_ngram = ngram(text=open('global_text.txt').read())
        self.population: List[Chromosome] = [
            Chromosome.random() for i in range(population_size)
        ]
        self.population_size: int = population_size
        self.fitness_cache = {}

    def calculate_fitness(self, chromosome: Chromosome):
        mapping = str(chromosome)
        if mapping in self.fitness_cache:
            return self.fitness_cache.get(mapping)

        fitness = 0.0
        decoded_ngram = ngram(
            text=chromosome.decode(self.encoded_text).lower()
        )
        for pair, occurrences in decoded_ngram.items():
            fitness += occurrences * log2(self.ref_ngram[pair] or 1)

        self.fitness_cache[mapping] = fitness
        return fitness

    def decode(self) -> str:
        found = False
        generation = 0
        start = time()

        repeat_count = 0
        best_fitness = 0

        while True:
            population = sorted(
                self.population,
                key=self.calculate_fitness,
                reverse=True
            )

            best_chromosome = population[0]
            current_best_fitness = self.calculate_fitness(best_chromosome)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                repeat_count = 0
            elif current_best_fitness == best_fitness:
                repeat_count += 1

            print(generation, current_best_fitness, best_chromosome)
            if repeat_count > MAX_REPEAT_COUNT:
                print("Key:", best_chromosome)
                print("Elapsed Time:", time() - start)
                return best_chromosome.decode(self.encoded_text)

            self.population = []
            for i, chromosome in enumerate(population):
                if len(self.population) >= self.population_size:
                    break

                if i < (self.population_size / 10):
                    self.population.append(chromosome)
                else:
                    father, mother = sample(population[:20], 2)
                    boy, girl = Chromosome.crossover(father, mother)
                    self.population.append(boy)
                    self.population.append(girl)

            generation += 1
