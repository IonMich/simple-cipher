#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:58:11 2018

@author: yannis
"""

from collections import Counter
import numpy as np

## Uncomment the following for hipergator submissions
#import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt
from re import sub
from itertools import product
import string
import random
from operator import itemgetter


def generate_key(stringOfSymbols):
    """Takes a string of symbols and creates a one-by-one mapping in that space and its inverse
    
    returns the tuple of these mappings (dictionary instances) 
    which can be used as encryption-decryption Keys    
    """
    jumbled = ''.join(random.sample(stringOfSymbols,len(stringOfSymbols)))
    decryptionKey = dict(zip(jumbled,stringOfSymbols))
    encryptionKey = dict(zip(stringOfSymbols , jumbled))
    
    return decryptionKey , encryptionKey

def generate_random_from_phrase(phrase,encryptKey):
    """Takes a phrase (string) and an encryption key (dictionary) as arguments
    
    returns the jumbled phrase as a string after encrypting the phrase    
    """
    jumbled_phrase = ""
    for symbol in phrase:
        jumbled_phrase += encryptKey[symbol]
    
    return jumbled_phrase

def decrypt_with_key(jumbled,decryptKey):
    """Takes a jumbled phrase (string) and a decryption key (dictionary) as arguments
    
    returns the decrypted phrase as a string    
    """
    decryptedPhrase = ""
    for symbol in jumbled:
        decryptedPhrase += decryptKey[symbol]
    
    return decryptedPhrase

     

def unigramGuess(jumbled,theUnigramRef):
    """Take a jumbled phrase and using the Reference Text, 
    
    return a unigram guess key
    """
    jumbledCounts = Counter(jumbled)
    for onegram in abcSpace:
        jumbledCounts[onegram] += 0
    sortedRefList = sorted(theUnigramRef, key=theUnigramRef.get, reverse=True)
    
    cleverKey = {}
    i = 0
    for w in sorted(jumbledCounts, key=jumbledCounts.get, reverse=True):
        
        cleverKey[w] = sortedRefList[i]
        i += 1
    
    return cleverKey

###### No longer useful code
#def bigram_generator():
#    """
#    
#    
#    """
#    if bool(bigramsRef):
#        for keys in bigramsRef:
#            yield keys
#    else:
#        for letters in product(abcSpace, repeat=2):
#            yield ''.join(letters)

#def occurrences(string, sub):
#    """
#    https://stackoverflow.com/questions/2970520/string-count-with-overlapping-occurrences
#    
#    string.count(substring) counts only once overlaping occurrences
#    """
#    count = start = 0
#    while True:
#        start = string.find(sub, start) + 1
#        if start > 0:
#            count+=1
#        else:
#            return count

#def occurrences(string, sub):
#    """
#    https://stackoverflow.com/questions/2970520/string-count-with-overlapping-occurrences
#    
#    string.count(substring) counts only once overlaping occurrences
#    """
#    return string.count(sub)

    
#def score_function123(dataCounter,theBigramsRef,theTrigramsRef,theDecryptedUnigramCounts,theDecryptedBigramCounts,theDecryptedTrigramCounts,normalization=1):
#    """
#    
#    OLD!!!! don't use again. Numpy array instead of loop makes the computation extremely efficient 
#    """
#    score = 0
#    
#    for unigram in n_gram_generator(1):
#        score += theDecryptedUnigramCounts[unigram] * np.log(dataCounter[unigram])
#    for bigram in bigram_generator():
#        score += theDecryptedBigramCounts[bigram] * np.log(theBigramsRef[bigram])
#    for trigram in n_gram_generator(3):
#        score += theDecryptedTrigramCounts[trigram] * np.log(theTrigramsRef[trigram])
#    
#    return score / normalization   


#def find_N_gramCounts(jumbled,theKey,n,addOne=False):
#    """
#    not useful anymore
#    """
#    theText = decrypt_with_key(jumbled,theKey)
#    theNgramCounts = {}
#    for Ngram in n_gram_generator(n):
#            theNgramCounts[Ngram] = occurrences(theText,Ngram)
#            if addOne:
#                theNgramCounts[Ngram] += 1
#    return theNgramCounts      

#def score_function123(jumbled,theKey,theUniBool=True,theBiBool=True,theTriBool=True,normalization=1):
#    """
#    float 32
#    In Python 3.6 dictionaries are ordered
    
#    OLD!!!! don't use for small phrases. unigramCounts is mostly empty so this method is inefficient for small strings
#    """
#    
#    score = 0
#    if theUniBool:
#        theUnigramCounts = find_N_gramCounts(jumbled,theKey,1)
#        uniLog = np.log(np.fromiter(unigramsRef.values(),dtype=np.float32))
#        uniCounts = np.fromiter(theUnigramCounts.values(),dtype=np.float32)
#        score += uniLog.dot(uniCounts)       
#    if theBiBool:
#        theBigramCounts = find_N_gramCounts(jumbled,theKey,2)
#        biLog = np.log(np.fromiter(bigramsRef.values(),dtype=np.float32))
#        biCounts = np.fromiter(theBigramCounts.values(),dtype=np.float32)
#        score += biLog.dot(biCounts)
#    if theTriBool:
#        theTrigramCounts = find_N_gramCounts(jumbled,theKey,3)
#        triLog = np.log(np.fromiter(trigramsRef.values(),dtype=np.float32))
#        triCounts = np.fromiter(theTrigramCounts.values(),dtype=np.float32)
#        score += triLog.dot(triCounts)
#    
#    return score / normalization  

def score_function123(jumbled,theKey,theUniBool=True,theBiBool=True,theTriBool=True,normalization=1):
    """Take a jumbled phrase (string) and a (possibly tentative) decryption key and 
    
    
    
    return the log score (float) corresponding to the uni/bi/tri-gram counts on the reference text
    the booleans determine which score functions to use
    
    for small encrypted text lengths this should go much faster than my previous implementations, 
    for very large cipher texts it might become inefficient
    """
    
    score = 0
    if theUniBool:
        oneCounter = find_N_gramCounts_forScore(jumbled,theKey,1)
        for unigram , counts  in oneCounter.items():
            score += counts * np.log(unigramsRef[unigram]) 
    if theBiBool:
        twoCounter = find_N_gramCounts_forScore(jumbled,theKey,2)
        for bigram , counts  in twoCounter.items():
            score += counts * np.log(bigramsRef[bigram])
    if theTriBool:
        threeCounter = find_N_gramCounts_forScore(jumbled,theKey,3)
        for trigram , counts  in threeCounter.items():
            score += counts * np.log(trigramsRef[trigram])
    
    return score / normalization
    

def chunk_string(theString, n):
    """Cut a string into n-letter overlapping chunks and 
    
    return a list of these n-letter chunks 
    """
    return [theString[i:i+n] for i in range(len(theString)-n+1)]


def find_N_gramCounts_forScore(jumbled,theKey,n):
    """Find the counts of a jumbled phrase by first deciphering with a key and then cutting it in chunks
    
    Returns the Counter of the chunks
    """
    theText = decrypt_with_key(jumbled,theKey)
    chunked = chunk_string(theText, n)
    return Counter(chunked)
        

def n_gram_generator(n):
    """Creates an iterator over all possible n-letter (or space) ordered combinations"""
    
    if n == 2 and len(bigramsRef) == len(abcSpace)**2:
        for keys in bigramsRef:
            yield keys
    elif n == 3 and bool(trigramsRef) == len(abcSpace)**3:
        for keys in trigramsRef:
            yield keys        
    else:
        for letters in product(abcSpace, repeat=n):
            yield ''.join(letters)   


def find_N_gramCounts_forReference(refText,nList,addOne=True):
    """Find n-gram counts from a reference text
    
    Returns a generator of over the elements of nList    
    """
    for n in nList:
        chunked = chunk_string(refText, n)
        chunkedCounter = Counter(chunked)
        for nGram in n_gram_generator(n):
            if addOne:
                chunkedCounter[nGram] += 1
            else:
                ## generate all keys with zero counts                
                chunkedCounter[nGram] += 0
        yield chunkedCounter
    

def accuracy(decryptedText,originalPhrase):
    """calculates the coincidence between a decipheredText and an originalPhrase
    
    Returns the ratio (float) of correct distinct letters divided by the total number of distinct letters in the phrase
    """
    distinctLetters = 0
    correctLetters = 0
    for symbol in abcSpace:
        pos = originalPhrase.find(symbol)
        if pos != -1:
            distinctLetters += 1
            if decryptedText[pos] == symbol:
                correctLetters += 1
    return correctLetters / distinctLetters


def MCMC_NoAnnealingDecryptor(jumbled,initGuessKey,maxIters=10000,
                                        keepBest=True,oneBool=False,
                                        biBool=False,triBool=False,metropolis=True,wordSearch=True):
    """This MCMC decryptor takes as argument a jumbled phrase from a substitution cipher an initial guess key,
    
    and returns a decryption key (dictionary) after a set number of iterations (maxIters)
    
    As additional optional parameters, one can choose
    - if the program returns the best or the last result
    - whether the score functions will take into account uni/bi/tri-gram counts
    - whether to do word search in the reference text
    - whether to implement the general metropolis acceptance rule (you should)
    """

    bestKeys = [({},0)]*10
        
    currentKey = initGuessKey.copy()
    
    currentScore = score_function123(jumbled,currentKey,oneBool,biBool,triBool)

    proposedKey = currentKey.copy()
    
    
    ## initialization of score evolution wrt jj
    ## jj increases by 1 every 100 counts
    jj = 0
    scoreEvolution = np.zeros((2,300000))
    
    currentText = decrypt_with_key(jumbled,currentKey)
    
    for i in range(maxIters):
        
        
        if i%1000 == 0 :
            
            scoreEvolution[:,jj] = i , currentScore
            jj += 1
        
        aKey , bKey = random.choices(abcSpace,k=2)
        proposedKey[aKey], proposedKey[bKey] = currentKey[bKey], currentKey[aKey]
        
        proposedScore = score_function123(jumbled,proposedKey,oneBool,biBool,triBool)
        
        deltaScore = proposedScore - currentScore
        if metropolis == False and deltaScore < 0 :
            continue
            
        if np.exp( deltaScore )  > random.random():
            currentKey[aKey], currentKey[bKey] = currentKey[bKey], currentKey[aKey]
            currentScore = proposedScore
            
            if keepBest and currentScore > bestKeys[-1][1]:
                if any(aBestScore == currentScore for _ , aBestScore in bestKeys):
                    pass
                else:        
                    bestKeys[-1] = (currentKey.copy(),currentScore)
                    bestKeys.sort(key=itemgetter(1), reverse=True )   
#                    print("Key Added! I got: {}\tAnd Score: {}".format(decrypt_with_key(jumbled,currentKey),currentScore))
                    if wordSearch == True:
                        currentText = decrypt_with_key(jumbled,currentKey)
                        wordsInText = currentText.split(" ")
                        while '' in wordsInText:
                            wordsInText.remove('')
                        ## check if all words of the phrase and in the reference text
                        ## if they are, complete the chain (essentially assign score of infinity)
                        for word in wordsInText:
                            if read_data.count(word.center(len(word)+2)) > 0 :
                                pass
                            else:
                                break
                        else:
#                            print("I found something!")
#                            print(currentText)
                            return currentKey.copy(), currentScore , scoreEvolution[:,:jj]
                    

        else:
            proposedKey[aKey], proposedKey[bKey] = currentKey[aKey], currentKey[bKey]
       
#    print("Finished the chain at i:",i)
    
    if keepBest:
        currentText = decrypt_with_key(jumbled,bestKeys[0][0])
#        print(bestKeys[0][1])
#        print(currentText)
        return bestKeys[0][0].copy(), currentScore , scoreEvolution[:,:jj]
    else:
        currentText = decrypt_with_key(jumbled,currentKey)
#        print(currentScore)
#        print(currentText)
        return currentKey.copy(), currentScore , scoreEvolution[:,:jj]


def MCMC_annealing123Decryptor_keepBest(jumbled,initGuessKey,
                                        Tmax=5,Tmin=5E-1,tau=1E4,
                                        keepBest=True,oneBool=False,biBool=False,triBool=False,wordSearch=True):
    """
    This MCMC decryptor takes as argument a jumbled phrase from a substitution cipher an initial guess key,
    three parameters for the annealing (max Temp, min Temp, \tau)
    
    and returns a decryption key (dictionary)
    As additional optional parameters, one can choose
    - if the program returns the best or the last result
    - whether the score functions will take into account uni/bi/tri-gram counts
    - whether to do word search in the reference text    
    """
    
    bestKeys = [({},0)]*10
        
    currentKey = initGuessKey.copy()
    
    currentScore = score_function123(jumbled,currentKey,oneBool,biBool,triBool)

    proposedKey = currentKey.copy()
    
    t = 0
    T = Tmax
    
    ## initialization of score evolution wrt T 
    jj = 0
    scoreEvolution = np.zeros((2,300000))
    
    currentText = decrypt_with_key(jumbled,currentKey)
    
    while T>Tmin:

        # Cooling
        t += 1
        T = Tmax * np.exp(-t/tau)
        
        
        if t%100 == 0 :
            
            scoreEvolution[:,jj] = T , currentScore
            jj += 1
        
        aKey , bKey = random.choices(abcSpace,k=2)
        proposedKey[aKey], proposedKey[bKey] = currentKey[bKey], currentKey[aKey]
        
        proposedScore = score_function123(jumbled,proposedKey,oneBool,biBool,triBool)
        
        deltaScore = proposedScore - currentScore
        if np.exp( deltaScore / T)  > random.random():
            currentKey[aKey], currentKey[bKey] = currentKey[bKey], currentKey[aKey]
            currentScore = proposedScore
            
            if keepBest and currentScore > bestKeys[-1][1]:
                if any(aBestScore == currentScore for _ , aBestScore in bestKeys):
                    pass
                else:        
                    bestKeys[-1] = (currentKey.copy(),currentScore)
                    bestKeys.sort(key=itemgetter(1), reverse=True )   
#                    print("Key Added! I got: {}\tAnd Score: {}".format(decrypt_with_key(jumbled,currentKey),currentScore))
                    
                    if wordSearch ==True:
                        currentText = decrypt_with_key(jumbled,currentKey)
                        wordsInText = currentText.split(" ")
                        while '' in wordsInText:
                            wordsInText.remove('')
                        ## check if all words of the phrase and in the reference text
                        ## if they are, complete the chain (essentially assign score of infinity)
                        for word in wordsInText:
                            if read_data.count(word.center(len(word)+2)) > 0 :
                                pass
                            else:
                                break
                        else:
                            print("I found something!")
                            print(currentText)
                            return currentKey.copy(), currentScore , scoreEvolution[:,:jj]
                    
        else:
            proposedKey[aKey], proposedKey[bKey] = currentKey[aKey], currentKey[bKey]
       
#    print("exit t:",t)
    
    if keepBest:
        currentText = decrypt_with_key(jumbled,bestKeys[0][0])
#        print(currentText)
#        print(bestKeys[0][1])        
        return bestKeys[0][0].copy() , currentScore , scoreEvolution[:,:jj]
    else:
        currentText = decrypt_with_key(jumbled,currentKey)
#        print(currentText)
#        print(currentScore)
        return currentKey.copy(), currentScore , scoreEvolution[:,:jj]
    
    

## Create a string with all lowercase letters and space at the end
abcSpace = ''.join((string.ascii_lowercase,' '))

## Identity Key {"a":"a","b":"b",....}
identityKey = dict(zip(abcSpace,abcSpace))

## Create two encryption-decryption key pairs for usage later
myDecryptKey , myEncryptKey = generate_key(abcSpace)
myDecryptKey2 , myEncryptKey2 = generate_key(abcSpace)

## Read the reference text
with open('wp.txt') as f:
    ## read the TXT file and convert it to lower case
    read_data = f.read().lower()
    ## convert all non-alphbetical characters to space
    read_data = sub('[^a-z]+', ' ', read_data)
    readDataCounter = Counter(read_data)
    print(Counter(read_data))

### writing to a text file    
#with open("Output.txt", "w") as text_file:
#    print("{}".format(read_data[:1000]), file=text_file)


## Compute the reference uni/bi/tri-gram counts 
bigramsRef = {}
trigramsRef = {}
NgramsRef = find_N_gramCounts_forReference(read_data,[1,2,3])
unigramsRef =  next(NgramsRef)
bigramsRef =  next(NgramsRef)
trigramsRef =  next(NgramsRef)

## as we discuss in the report we can justify adding spaces
## at the beginning and at the end of this particular phrase
myPhrase = " the answer to life the universe and everything is forty two "

#myPhrase = read_data[20005:20075]

## Encrypt the phrase with a substitution cipher
mixedUp = generate_random_from_phrase(myPhrase,myEncryptKey)

## Initial Unigram Guess
myInitGuessKey = unigramGuess(mixedUp,unigramsRef)


## Deciphered texts using different Decryption Keys
decryptedText = decrypt_with_key(mixedUp,myDecryptKey)
decryptedText2 = decrypt_with_key(mixedUp,myDecryptKey2) ## incorrect decryption to check score
decryptedTextUni = decrypt_with_key(mixedUp,myInitGuessKey)
   
## Computing Bigram and Trigram Scores
myScore = score_function123(mixedUp,myDecryptKey,False,True,False) 
myScore_2 = score_function123(mixedUp,myDecryptKey2,False,True,False) 
myScore123 = score_function123(mixedUp,myDecryptKey,False,False,True) 
myScore123_2 = score_function123(mixedUp,myDecryptKey2,False,False,True) 


## Sanity checks
print("Original Text:",myPhrase)
print("Mixed Up:",mixedUp)
print("Decrypted Text with Key:",decryptedText)
print("Decrypted Text with Key2:",decryptedText2)
print("Decrypted with Educaed Guess:",decryptedTextUni)
print("Score with Decryptor:",myScore)
print("Score with Decryptor2:",myScore_2)
print("Score123 with Decryptor:",myScore123)
print("Score123 with Decryptor2:",myScore123_2)


#########   TESTS


#numChains = 50
#accuracies = np.zeros(numChains)
#
### Check Evolution
#plt.figure(1)
#maxTemp = 5
#minTemp = 0.5
#tau = 1E5
#
### if no annealing:
#iterationsMax = 50000
#
##plt.xlim(maxTemp,minTemp)
#plt.xlim(0,20000)
#for i in range(numChains):
#    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey,
#                                                          keepBest=True,
#                                                          biBool=False,triBool=True)
#    myKeyGuess = keyFound
#    keyFound , _ , score_vs_T = MCMC_annealing123Decryptor_keepBest(mixedUp,myKeyGuess,
#                                                                    Tmax=maxTemp,Tmin=minTemp,tau=tau,
#                                                                    keepBest=True,biBool=False,triBool=True)
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
#    
#    accuracies[i] = accuracy(textFound,myPhrase)
#    
#print("Correct Bigram Score:",myScore)
#print("Correct Trigram Score:",myScore123)
#print("Average Accuracy of Distinct Letters: {}".format(np.mean(accuracies)))
#print("Number of successes: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  


################################################
####### ( uncomment the segments to run tests )
########## TEST 1 
######## Metropolis vs Only Positive

#numChains = 50
#accuracies = np.zeros(numChains)
#accuraciesMetro = np.zeros(numChains)
### Check Evolution
#
#
### no annealing:
#iterationsMax = 20000
#plt.xlim(0,iterationsMax)
#for i in range(numChains):
#    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myDecryptKey2, maxIters=iterationsMax,
#                                                          keepBest=False,metropolis=False,
#                                                          biBool=True,triBool=False)
#
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.figure(1)
#    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
#    plt.xlim(0,iterationsMax)
#    
#    accuracies[i] = accuracy(textFound,myPhrase)
#    
##    keyFound , _ , score_vs_T_Metro = MCMC_NoAnnealingDecryptor(mixedUp,myDecryptKey2,maxIters=iterationsMax,
##                                                          keepBest=False,metropolis=True,
##                                                          biBool=True,triBool=False)
##    textFound = decrypt_with_key(mixedUp,keyFound)
##    plt.figure(2)
##    plt.plot(score_vs_T_Metro[0,:],score_vs_T_Metro[1,:])
##    plt.xlim(0,iterationsMax)
##    
##    accuraciesMetro[i] = accuracy(textFound,myPhrase)
##
#print("Correct Bigram Score:",myScore)
##print("Correct Trigram Score:",myScore123)
#print("Average Accuracy of Distinct Letters without Metropolis: {}".format(np.mean(accuracies)))
#print("Average Accuracy of Distinct Letters with Metropolis: {}".format(np.mean(accuraciesMetro)))
#print("Number of successes without Metropolis: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  
#print("Number of successes with Metropolis: {} out of {}".format(np.count_nonzero(accuraciesMetro==1),numChains))  


######## TEST 2
######## Keep Best and unigram guess

#numChains = 50
#accuracies = np.zeros(numChains)
#accuraciesBestInit = np.zeros(numChains)
### Check Evolution
#
#
### no annealing:
#iterationsMax = 20000
#plt.xlim(0,iterationsMax)
#for i in range(numChains):
#    myDecryptKey2 , myEncryptKey2 = generate_key(abcSpace)
#    print("\nIteration {} out of {}\n".format(i+1,numChains))
#    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myDecryptKey2, maxIters=iterationsMax,
#                                                          keepBest=False,
#                                                          biBool=True,triBool=False)
#
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.figure(1)
#    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
#    plt.xlim(0,iterationsMax)
#    
#    accuracies[i] = accuracy(textFound,myPhrase)
#    
#    keyFound , _ , score_vs_T_BestInit = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey,maxIters=iterationsMax,
#                                                          keepBest=True,
#                                                          biBool=True,triBool=False)
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.figure(2)
#    plt.plot(score_vs_T_BestInit[0,:],score_vs_T_BestInit[1,:])
#    plt.xlim(0,iterationsMax)
#    
#    accuraciesBestInit[i] = accuracy(textFound,myPhrase)
#
#print("Correct Bigram Score:",myScore)
##print("Correct Trigram Score:",myScore123)
#print("Average Accuracy of Distinct Letters without BestKeep / good guess: {}".format(np.mean(accuracies)))
#print("Average Accuracy of Distinct Letters with BestKeep / good guess: {}".format(np.mean(accuraciesBestInit)))
#print("Number of successes without BestKeep / good guess: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  
#print("Number of successes with BestKeep / good guess: {} out of {}".format(np.count_nonzero(accuraciesBestInit==1),numChains))  




######## TEST 3
######## changing the number of iterations

#numChains = 50
#accuracies = np.zeros(numChains)
#accuracies10 = np.zeros(numChains)
### Check Evolution
#
#
#### no annealing:
##iterationsMax = 20000
##for i in range(numChains):
##    print("\nIteration {} out of {}\n".format(i+1,numChains))
##    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey, maxIters=iterationsMax,
##                                                          keepBest=True,
##                                                          biBool=True,triBool=False)
##
##    textFound = decrypt_with_key(mixedUp,keyFound)
##    plt.figure(1)
##    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
##    plt.xlim(0,iterationsMax)
##    
##    accuracies[i] = accuracy(textFound,myPhrase)
#    
### no annealing:
#iterationsMax = 5000
#for i in range(numChains):
#    print("\nIteration {} out of {}\n".format(i+1,numChains))
#    keyFound , _ , score_vs_T_10 = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey,maxIters=iterationsMax,
#                                                          keepBest=True,
#                                                          biBool=True,triBool=False)
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.figure(2)
#    plt.plot(score_vs_T_10[0,:],score_vs_T_10[1,:])
#    plt.xlim(0,iterationsMax)
#    
#    accuracies10[i] = accuracy(textFound,myPhrase)
#
#print("Correct Bigram Score:",myScore)
##print("Correct Trigram Score:",myScore123)
#print("Average Accuracy of Distinct Letters without 50k: {}".format(np.mean(accuracies)))
#print("Average Accuracy of Distinct Letters with 10k: {}".format(np.mean(accuracies10)))
#print("Number of successes without 50k: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  
#print("Number of successes with 10k: {} out of {}".format(np.count_nonzero(accuracies10==1),numChains)) 




######### TEST 4
######### Tmax Tmin
#
#numChains = 50
#accuracies = np.zeros(numChains)
#
### Check Evolution
#
#maxTemp = 30
#minTemp = 0.1
#tau = 1E4
#
#plt.figure(1)
#plt.xlim(maxTemp,minTemp)
#
#for i in range(numChains):
#    print("\nIteration {} out of {}\n".format(i+1,numChains))
##    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey,
##                                                          keepBest=True,
##                                                          biBool=False,triBool=True)
##    myKeyGuess = keyFound
#    myKeyGuess = myInitGuessKey
#    keyFound , _ , score_vs_T = MCMC_annealing123Decryptor_keepBest(mixedUp,myKeyGuess,
#                                                                    Tmax=maxTemp,Tmin=minTemp,tau=tau,
#                                                                    keepBest=True,biBool=True,triBool=False)
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
#    
#    accuracies[i] = accuracy(textFound,myPhrase)
#    
#print("Correct Bigram Score:",myScore)
##print("Correct Trigram Score:",myScore123)
#print("Average Accuracy of Distinct Letters: {}".format(np.mean(accuracies)))
#print("Number of successes: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  



######### TEST 5
######### changing tau
#
#numChains = 50
#accuracies = np.zeros(numChains)
#
### Check Evolution
#
#maxTemp = 5
#minTemp = 0.1
#tau = 1E5
#
#plt.figure(1)
#plt.xlim(maxTemp,minTemp)
#
#for i in range(numChains):
#    print("\nIteration {} out of {}\n".format(i+1,numChains))
##    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey,
##                                                          keepBest=True,
##                                                          biBool=False,triBool=True)
##    myKeyGuess = keyFound
#    myKeyGuess = myInitGuessKey
#    keyFound , _ , score_vs_T = MCMC_annealing123Decryptor_keepBest(mixedUp,myKeyGuess,
#                                                                    Tmax=maxTemp,Tmin=minTemp,tau=tau,
#                                                                    keepBest=True,biBool=True,triBool=False)
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
#    
#    accuracies[i] = accuracy(textFound,myPhrase)
#    
#ax = plt.gca()
#ax.set_xscale('log')
#plt.axhline(y=myScore, color='k', linestyle='-')
#textstr = '\n'.join((
#        "Mean Accuracy: {:.3f}".format(np.mean(accuracies)),
#        "Successes: {}/{}".format(np.count_nonzero(accuracies==1),numChains)))  
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#
## place a text box in upper left in axes coords
#plt.text(0.50, 0.20, textstr, transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#plt.xlabel("Temperature")
#plt.ylabel("bigram score function")
#plt.title(r"$\tau = $")
#plt.ylim(450,630)
#
#
#print("Correct Bigram Score:",myScore)
##print("Correct Trigram Score:",myScore123)
#print("Average Accuracy of Distinct Letters: {}".format(np.mean(accuracies)))
#print("Number of successes: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  
#



######### TEST 6
######### trigrams
#
#numChains = 50
#accuracies = np.zeros(numChains)
#
### Check Evolution
#
#maxTemp = 5
#minTemp = 0.1
#tau = 1E5
#
#plt.figure(1)
#plt.xlim(maxTemp,minTemp)
#
#for i in range(numChains):
#    print("\nIteration {} out of {}\n".format(i+1,numChains))
##    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey,
##                                                          keepBest=True,
##                                                          biBool=False,triBool=True)
##    myKeyGuess = keyFound
#    myKeyGuess = myInitGuessKey
#    keyFound , _ , score_vs_T = MCMC_annealing123Decryptor_keepBest(mixedUp,myKeyGuess,
#                                                                    Tmax=maxTemp,Tmin=minTemp,tau=tau,
#                                                                    keepBest=True,biBool=False,triBool=True,wordSearch=False)
#    textFound = decrypt_with_key(mixedUp,keyFound)
#    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
#    
#    accuracies[i] = accuracy(textFound,myPhrase)
#    
#ax = plt.gca()
#ax.set_xscale('log')
#plt.axhline(y=myScore123, color='k', linestyle='-')
#textstr = '\n'.join((
#        "Mean Accuracy: {:.3f}".format(np.mean(accuracies)),
#        "Successes: {}/{}".format(np.count_nonzero(accuracies==1),numChains)))  
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#
## place a text box in upper left in axes coords
#plt.text(0.50, 0.20, textstr, transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#plt.xlabel("Temperature")
#plt.ylabel("Trigram score function")
#plt.title(r"$\tau = 10^5$")
#plt.ylim(250,500)
#
#
##print("Correct Bigram Score:",myScore)
#print("Correct Trigram Score:",myScore123)
#print("Average Accuracy of Distinct Letters: {}".format(np.mean(accuracies)))
#print("Number of successes: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  


######## TEST 7
######## including word search

numChains = 10
accuracies = np.zeros(numChains)

## Check Evolution

maxTemp = 5
minTemp = 0.1
tau = 1E5

plt.figure(1)
plt.xlim(maxTemp,minTemp)
#### no annealing:
iterationsMax = 10000

for i in range(numChains):
    print("\nIteration {} out of {}\n".format(i+1,numChains))
    keyFound , _ , score_vs_T = MCMC_NoAnnealingDecryptor(mixedUp,myInitGuessKey,maxIters=iterationsMax,
                                                          keepBest=True,
                                                          biBool=True,triBool=False)
    myKeyGuess = keyFound
    keyFound , _ , score_vs_T = MCMC_annealing123Decryptor_keepBest(mixedUp,myKeyGuess,
                                                                    Tmax=maxTemp,Tmin=minTemp,tau=tau,
                                                                    keepBest=True,biBool=False,triBool=True,wordSearch=True)
    textFound = decrypt_with_key(mixedUp,keyFound)
    plt.plot(score_vs_T[0,:],score_vs_T[1,:])
    
    accuracies[i] = accuracy(textFound,myPhrase)
    
ax = plt.gca()
ax.set_xscale('log')
plt.axhline(y=myScore123, color='k', linestyle='-')
textstr = '\n'.join((
        "Mean Accuracy: {:.3f}".format(np.mean(accuracies)),
        "Successes: {}/{}".format(np.count_nonzero(accuracies==1),numChains)))  
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
plt.text(0.50, 0.20, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.xlabel("Temperature")
plt.ylabel("Trigram score function")
plt.title(r"$\tau = 10^5$")
plt.ylim(250,500)
#plt.savefig("test7.png")

#print("Correct Bigram Score:",myScore)
print("Correct Trigram Score:",myScore123)
print("Average Accuracy of Distinct Letters: {}".format(np.mean(accuracies)))
print("Number of successes: {} out of {}".format(np.count_nonzero(accuracies==1),numChains))  
