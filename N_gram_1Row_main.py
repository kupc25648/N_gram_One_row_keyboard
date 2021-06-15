'''
==============================================================================================================
Import
==============================================================================================================
'''
import pandas as pd
import numpy as np
import random
import string

'''
==============================================================================================================
Preparing data
==============================================================================================================
'''

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
key = ['1','2','3','4','5','6','7','8','9','10']

data = 'data/Warandpeace.txt'
#data = 'data/alice.txt'
with open(data, 'r') as file:
    alice = file.read().replace('\n', '')

alice = alice.translate(str.maketrans('', '', string.punctuation))
alice = alice.lower()
alice = alice.split()

random.shuffle(alice)
s_train = alice[:-int(len(alice)/10000)]
s_test = alice[-int(len(alice)/10000):]

'''
==============================================================================================================
Making corpus

iterate word in source
    iterate alphabet in word
        add the key before the alphabet
        add modified word into corpus
==============================================================================================================
'''
def add_to_corpus(word,corpus):
    for i in range(len(word)):
        new_world = '_'
        new_world+=word[:i]
        if (word[i] == "q") or (word[i] == "a") or (word[i] == "z"):
            new_world+=str(1)
        elif (word[i] == "w") or (word[i] == "s") or (word[i] == "x"):
            new_world+=str(2)
        elif (word[i] == "e") or (word[i] == "d") or (word[i] == "c"):
            new_world+=str(3)
        elif (word[i] == "r") or (word[i] == "f") or (word[i] == "v"):
            new_world+=str(4)
        elif (word[i] == "t") or (word[i] == "g") or (word[i] == "b"):
            new_world+=str(5)
        elif (word[i] == "y") or (word[i] == "h") or (word[i] == "n"):
            new_world+=str(6)
        elif (word[i] == "u") or (word[i] == "j") or (word[i] == "m"):
            new_world+=str(7)
        elif (word[i] == "i") or (word[i] == "k"):
            new_world+=str(8)
        elif (word[i] == "o") or (word[i] == "l"):
            new_world+=str(9)
        elif (word[i] == "p"):
            new_world+=str(0)
        new_world+=word[i:]
        new_world = new_world[1:]
        corpus.append(new_world)



'''
==============================================================================================================
Making N-gram function

iterate word in corpus
try: making 2 gram function
try: making 3 gram function
try: making 4 gram function
try: making 5 gram function
try: making 6 gram function
==============================================================================================================
'''
def add_2_gram(word,my_2_gram):
    for i in range(len(word)):
        try:
            if type(int(word[i]))==int:
                my_2_gram[int(word[i])][alphabet.index(word[i+1])] += 1
        except:
            pass

def add_3_gram(word,my_3_gram):
    for i in range(len(word)):
        try:
            if type(int(word[i]))==int:
                my_3_gram[int(word[i])][alphabet.index(word[i-1])][alphabet.index(word[i+1])] += 1
        except:
            pass

def add_4_gram(word,my_4_gram):
    for i in range(len(word)):
        try:
            if type(int(word[i]))==int:
                my_4_gram[int(word[i])][alphabet.index(word[i-1])][alphabet.index(word[i-2])][alphabet.index(word[i+1])] += 1
        except:
            pass

def add_5_gram(word,my_5_gram):
    for i in range(len(word)):
        try:
            if type(int(word[i]))==int:
                my_5_gram[int(word[i])][alphabet.index(word[i-1])][alphabet.index(word[i-2])][alphabet.index(word[i-3])][alphabet.index(word[i+1])] += 1
        except:
            pass

def add_6_gram(word,my_6_gram):
    for i in range(len(word)):
        try:
            if type(int(word[i]))==int:
                my_6_gram[int(word[i])][alphabet.index(word[i-1])][alphabet.index(word[i-2])][alphabet.index(word[i-3])][alphabet.index(word[i-4])][alphabet.index(word[i+1])] += 1
        except:
            pass
'''
==============================================================================================================
Preparing N-gram
==============================================================================================================
'''
base = s_train
corpus = []

# last dim is the result
my_2_gram = np.zeros((10,26))
my_3_gram = np.zeros((10,26,26))
my_4_gram = np.zeros((10,26,26,26))
my_5_gram = np.zeros((10,26,26,26,26))
my_6_gram = np.zeros((10,26,26,26,26,26))
for i in range(len(base)):
    add_to_corpus(base[i],corpus)

for i in range(len(corpus)):
    add_2_gram(corpus[i],my_2_gram)
    add_3_gram(corpus[i],my_3_gram)
    add_4_gram(corpus[i],my_4_gram)
    add_5_gram(corpus[i],my_5_gram)
    add_6_gram(corpus[i],my_6_gram)


'''
==============================================================================================================
Making prediction functions
==============================================================================================================
'''
def interpret_long(num):
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    key = ['1','2','3','4','5','6','7','8','9','10']
    result = '_'
    for i in range(len(num)):
        if i==0:
            # first alphabet use 2 gram
            result += alphabet[np.argmax(my_2_gram[int(num[i])])]
        if i==1:
            if int(my_3_gram[int(num[i])][alphabet.index(result[-1])][np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])]) != 0:
                result += alphabet[np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])] # second alphabet use 3 gram
            else:
                result += alphabet[np.argmax(my_2_gram[int(num[i])])] # second alphabet use 2 gram
        if i==2:
            if int(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])]) != 0:
                result += alphabet[np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])] # third alphabet use 4 gram
            else:
                if int(my_3_gram[int(num[i])][alphabet.index(result[-1])][np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])]) != 0:
                    result += alphabet[np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])] # third alphabet use 3 gram
                else:
                    result += alphabet[np.argmax(my_2_gram[int(num[i])])] # third alphabet use 2 gram

        if i==3:
            if int(my_5_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])][np.argmax(my_5_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])])]) != 0:
                result += alphabet[np.argmax(my_5_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])])] # fourth alphabet use 5 gram
            else:
                if int(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])]) != 0:
                    result += alphabet[np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])] # fourth alphabet use 4 gram
                else:
                    if int(my_3_gram[int(num[i])][alphabet.index(result[-1])][np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])]) != 0:
                        result += alphabet[np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])] # fourth alphabet use 3 gram
                    else:
                        result += alphabet[np.argmax(my_2_gram[int(num[i])])] # fourth alphabet use 2 gram

        if i>=4:
            if int(my_6_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])][alphabet.index(result[-4])][np.argmax(my_6_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])][alphabet.index(result[-4])])]) != 0:
                result += alphabet[np.argmax(my_6_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])][alphabet.index(result[-4])])] # fifth alphabet use 6 gram
            else:
                if int(my_5_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])][np.argmax(my_5_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])])]) != 0:
                    result += alphabet[np.argmax(my_5_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][alphabet.index(result[-3])])] # fifth alphabet use 5 gram
                else:
                    if int(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])]) != 0:
                        result += alphabet[np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])] # fifth alphabet use 4 gram
                    else:
                        if int(my_3_gram[int(num[i])][alphabet.index(result[-1])][np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])]) != 0:
                            result += alphabet[np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])] # fifth alphabet use 3 gram
                        else:
                            result += alphabet[np.argmax(my_2_gram[int(num[i])])] # fifth alphabet use 2 gram
    return result[1:]

def interpret_short(num):
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    key = ['1','2','3','4','5','6','7','8','9','10']
    result = '_'
    for i in range(len(num)):
        if i==0:
            # first alphabet use 2 gram
            result += alphabet[np.argmax(my_2_gram[int(num[i])])]
        if i==1:
            if int(my_3_gram[int(num[i])][alphabet.index(result[-1])][np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])]) != 0:
                result += alphabet[np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])] # second alphabet use 3 gram
            else:
                result += alphabet[np.argmax(my_2_gram[int(num[i])])] # second alphabet use 2 gram
        if i>=2:
            if int(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])][np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])]) != 0:
                result += alphabet[np.argmax(my_4_gram[int(num[i])][alphabet.index(result[-1])][alphabet.index(result[-2])])] # third alphabet use 4 gram
            else:
                if int(my_3_gram[int(num[i])][alphabet.index(result[-1])][np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])]) != 0:
                    result += alphabet[np.argmax(my_3_gram[int(num[i])][alphabet.index(result[-1])])] # third alphabet use 3 gram
                else:
                    result += alphabet[np.argmax(my_2_gram[int(num[i])])] # third alphabet use 2 gram


    return result[1:]

def interpret_zero(num):
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    key = ['1','2','3','4','5','6','7','8','9','10']
    result = '_'
    for i in range(len(num)):
        result += alphabet[np.argmax(my_2_gram[int(num[i])])]
    return result[1:]

'''
==============================================================================================================
Making test word functions (trun test into number)
==============================================================================================================
'''
def turn_to_num(word):
    string_num = '_'
    for i in range(len(word)):
        if (word[i] == "q") or (word[i] == "a") or (word[i] == "z"):
            string_num+=str(1)
        elif (word[i] == "w") or (word[i] == "s") or (word[i] == "x"):
            string_num+=str(2)
        elif (word[i] == "e") or (word[i] == "d") or (word[i] == "c"):
            string_num+=str(3)
        elif (word[i] == "r") or (word[i] == "f") or (word[i] == "v"):
            string_num+=str(4)
        elif (word[i] == "t") or (word[i] == "g") or (word[i] == "b"):
            string_num+=str(5)
        elif (word[i] == "y") or (word[i] == "h") or (word[i] == "n"):
            string_num+=str(6)
        elif (word[i] == "u") or (word[i] == "j") or (word[i] == "m"):
            string_num+=str(7)
        elif (word[i] == "i") or (word[i] == "k"):
            string_num+=str(8)
        elif (word[i] == "o") or (word[i] == "l"):
            string_num+=str(9)
        elif (word[i] == "p"):
            string_num+=str(0)
    return string_num[1:]


'''
==============================================================================================================
Test
==============================================================================================================
'''
correct_long = 0
wrong_long = 0
correct_long_len = np.zeros((1,50))
wrong_long_len = np.zeros((1,50))

correct_short = 0
wrong_short = 0
correct_short_len = np.zeros((1,50))
wrong_short_len = np.zeros((1,50))

correct_zero = 0
wrong_zero = 0
correct_zero_len = np.zeros((1,50))
wrong_zero_len = np.zeros((1,50))

all_test  = 0
for i in range(len(s_test)):
    test_num = turn_to_num(s_test[i])
    result_long = interpret_long(test_num)
    result_short = interpret_short(test_num)
    result_zero = interpret_zero(test_num)
    if s_test[i] == result_long:
        correct_long += 1
        correct_long_len[0][len(s_test[i])] += 1
    else:
        wrong_long += 1
        wrong_long_len[0][len(s_test[i])] += 1

    if s_test[i] == result_short:
        correct_short += 1
        correct_short_len[0][len(s_test[i])] += 1
    else:
        wrong_short += 1
        wrong_short_len[0][len(s_test[i])] += 1

    if s_test[i] == result_zero:
        correct_zero += 1
        correct_zero_len[0][len(s_test[i])] += 1
    else:
        wrong_zero += 1
        wrong_zero_len[0][len(s_test[i])] += 1

    all_test += 1

#Print accuray in each prediction methods
print('Long prediction  ----------------------')
print(correct_long/all_test)
print(correct_long_len)
print('Short prediction ----------------------')
print(correct_short/all_test)
print(correct_short_len)
print('Zero prediction  ----------------------')
print(correct_zero/all_test)
print(wrong_zero_len)


#Save csv for accuray in each prediction methods and word length
dataframe = pd.DataFrame(correct_long_len)
dataframe.to_csv('{}.csv'.format('correct_long_len'))
dataframe = pd.DataFrame(wrong_long_len)
dataframe.to_csv('{}.csv'.format('wrong_long_len'))

dataframe = pd.DataFrame(correct_short_len)
dataframe.to_csv('{}.csv'.format('correct_short_len'))
dataframe = pd.DataFrame(wrong_short_len)
dataframe.to_csv('{}.csv'.format('wrong_short_len'))

dataframe = pd.DataFrame(correct_zero_len)
dataframe.to_csv('{}.csv'.format('correct_zero_len'))
dataframe = pd.DataFrame(wrong_zero_len)
dataframe.to_csv('{}.csv'.format('wrong_zero_len'))
