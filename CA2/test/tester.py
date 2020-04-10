from code import Decoder
import sys, os
from time import time
import re
import numpy as np

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__


for i in range(2) :
    print('Test',i,'...')
    encoded_text = open("encoded_test_"+str(i)+".txt").read()

    blockPrint()
    start = time()
    d = Decoder(encoded_text)
    decoded_text = d.decode()
    enablePrint()
    print("\tElapsed Time:", time() - start)
    
    answer = open('decoded_test_'+str(i)+'.txt').read()
    words = np.array(answer.split())
    predicts = np.array(decoded_text.split())
    print('\tComplete answer =', 100 * (words==predicts).sum() / words.shape[0], '%')
    
    answer = re.sub('\W',' ',answer)
    decoded_text = re.sub('\W',' ',decoded_text)
    words = np.array(answer.split())
    predicts = np.array(decoded_text.split())
    print('\tWithout \\W answer =', 100 * (words==predicts).sum() / words.shape[0], '%')

    answer = answer.lower()
    decoded_text = decoded_text.lower()
    words = np.array(answer.split())
    predicts = np.array(decoded_text.split())
    print('\tLower case answer =', 100 * (words==predicts).sum() / words.shape[0], '%')