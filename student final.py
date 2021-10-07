#!/usr/bin/env python3
# COMP9444 Assignment 2
# Group Id: g023533
# Group Member:
# 	Yun Li, z5231701
# 	Tianchen Yang, z5248280
# Date: Nov / 2020

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
# import numpy as np
# import sklearn

from config import device

###############################
##### Answer of question  #####
###############################

'''
Q: Briefly describe how your program works, and explain any design and training decisions you made along the way.

How program works:

At very first, we need to turn the sentence into words for following operation and elinimating any useless information we got, 
then those words need to be preprocessing before we turn them into vectors due.
After that we use lstm with two extra linear layer to learn  
and we use corresponding loss function and weighting for the rating loss nad categpry loss. 
So this Models is basicly about keep learning the precise category and correct rating of an English reviews

Training decisions:
For epoch, lr and batchsize, we used the safest and perfect strategy we can think of, which is lower lr, smaller batchsize and higher epochs
Lower lr and smaller batchsize enable us to learn features carefully with low risk, 
but the process will be extremly slow with a too low lr so we used 0.008 to prevent that
And higher epoch gives us enough chance to get local minimum until we used 100% of potential of the current model and learning skill.

During the experiment, We reduced lr and batchsize in similar rate, 
and increase epoch until the loss stays in a small interval and doesn't decrease anymore

Obviously, LSTM is the best model to perform NLP for now, and we use multiple linear layer to be safe. 
(There are more discussion about network in following comments)

We tried use differnt avcivation and loss function for rate and category, 
it turns out combination of relu, which works as activation function, and entropy and nll, which works as loss function, yields higher score

About optimizer,At first, initial SGD optimizer is replaced by Adam optimizer since Adam has better perfermonce coveraging loss value at beggining, 
but SGD always outperform Adam with unknown reason, so we use SGD eventually.
We feel like Adam is very efficient but not accurate as SGD when the purpose is getting very high score 
'''


#######################################################
##### comments are added under the function name ######
#######################################################

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Any non-english character will be replace by space (' ') in new_sample, 
    (this can be seen as preprocessing but we do it here cause it's more convinient)
    and the space will be deleted by useing split function.
    we think this is the best way to keep the most features of the sentence
    """
    alphab = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    new_sample = ""
    for character in sample:
      if character in alphab:
        new_sample = new_sample + character
      else:
        new_sample = new_sample + ' '
        

    processed = new_sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    all the case is converted to lower case for the sake of word identification and word-to-vector transformation
    """
    new_sample = []
    for word in sample:
        word = word.lower()
        if len(word) > 0 and word not in stopWords: 
            new_sample.append(word)
    return new_sample

def postprocessing(batch, vocab):
    """
    Nothing here because we think over-processing may cause the reduction of dataset's features 
    """

    return batch

"""
To be general enough, we mixed multiple source of stop-words and deleted some of them
After that , we  add http, com, cn to remove the useless part of a url, 
because the content of the url should be enough to show the features
"""
stopWords =                   {'that', "shouldn't", 'do', 're', 'yourself', 'why', 'our', 'where', "he'll", 'as', 'herself', 'in', 
            'doesn', 'your', 've', 'needn', "should've", 'only', "don't", 'ours', 'through', 'any', 'and', 'by', 'those', 'ought',
            "they're", "didn't", 'com', "wouldn't", 'himself', 'an', 'not', "that'll", 'hers', 'http', 'them', "they'll", 'their', 
      "where's", 'd', 'above', 'on', 'ain', 'have', "we've", "here's", 'should', 'very', 'own', "haven't", 'his', "he's", 'other', 
       'is', 'my', 'then', "i've", 'what', "we're", 'itself', 'if', 'didn', 'when', "mustn't", "aren't", "why's", 'until', 'down', 
        'm', 'she', 'her', "won't", 'with', 'won', 'would', 'all', 'into', 'theirs', 'up', 'to', "he'd", "doesn't", 'at', 'yours', 
    "i'd", 'this', "shan't", "can't", 'll', 'its', 'weren', 'doing', 'o', "it's", 'than', "we'll", 'each', 'pm', "isn't", "how's", 
   'wasn', 'been', "they've", 'just', "i'll", 'aren', "hadn't", "there's", 'after', 'it', "they'd", 'was', 'has', 'so', "weren't", 
    'more', 'being', 'they', 'against', 'you', 't', 'between', 'there', 'are', "you're", 'him', 'am', 'further', 'gmail', 'about', 
   'because', 'during', 'y', "when's", 'were', 'again', 'some', 'here', 'below', 'the', 'does', 'for', 'themselves','no', "she's", 
       'did', "you'll", 'or', "let's", 'but', 'most', 'such', "that's", "she'll", 'he', 'now', 'a', 'shouldn', 's', 'how', 'whom', 
     "you've", 'which', "wasn't", 'mustn', 'haven', 'while', 'out', 'having', 'hadn', 'myself', 'these', 'same', "we'd", 'mightn', 
         'few', 'ourselves', 'too', "she'd", "who's", "couldn't", 'isn', 'shan', 'both', 'can', 'over', 'don', 'be', 'cn', 'once', 
     "hasn't", 'who', 'could', "i'm", 'i', 'had', 'we', 'before', "you'd", 'of', "what's", 'ma', 'nor', 'me', 'cannot', "needn't", 
                                             'hasn', 'from', 'off', "mightn't", 'wouldn', 'yourselves', 'under', 'will', 'couldn'}

"""
we used 300 to be more general 
"""                                             
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    For convertNetOutput, there is no complicated algorithms in this part. 
    Just seperate the rating to 0 or 1 and categories to 0, 1, 2, 3 or 4, 
    return as long type to ensure that our output should be simple and clear.
    To get the index of maximum, we used argmax and set dim arguments as 1.
    we also map the extrem value to maximum or minimum to be more general
    """
    #rating values between 0 and 1
    rating = ratingOutput.round()
    rating[rating > 1] = 1
    rating[rating < 0] = 0

    #category values between 0, 1, 2, 3, 4
    category = categoryOutput.round()
    category[category > 4] = 4
    category[category < 0] = 0
    #print(rating.long(), category.long())
    ratingOutput = torch.argmax(ratingOutput, dim=1)
    categoryOutput = torch.argmax(categoryOutput, dim=1)
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    We use Lstm plus two linear layer, and we use relu as the activation function.
    Btw, 300 inputs of lstm due to the dim of word vector
    we use output[:,-1,:] to get the final out put in the final timestep
    To show the effect of lstm throughly while not wasting too much nodes, 
    we used 300 inputs node at first hidden layer
    
    For networks, I choose Bi-direction LSTM network models in this text classification. First, Recurrent Neural Network
    is usually processing the word or text classification, and LSTM can better reduce gradient vanishing and exploding.
    using bi-LSTM is the combination of two LSTM,forwards and backwards, that have a reliable score weight compared to 
    other RNN models in this case(Get this conclusion after training hundreds of time using dozens of RNN models,
    ML and DL needs patience and luck). The parameter in this models affect sightly about score weight, so choosing a 
    suitable RNN models in extremely important.
    """
    def __init__(self):
        super(network, self).__init__()
        self.fc1 = tnn.Linear(300,150)
        self.fc2 = tnn.Linear(150,30)
        self.cate = tnn.Linear(30,5)
        self.rate = tnn.Linear(30,2)
        self.lstm = tnn.LSTM(300, 150, 1, batch_first = True, bidirectional = True)

    """
    It seems like use of log_softmax doesn't affect the result too much due to the loss function we used,
    but we sill keep them just to be safe, I think the relu is the most suitable activation here,
    and our previous experiment also proves that
    """

    def forward(self, input, length):
        output, (h_n, c_n) = self.lstm(input)
        mid1 = F.relu(self.fc1(output[:,-1,:]))
        mid2 = F.relu(self.fc2(mid1))
        return F.log_softmax(self.rate(mid2),dim = 1), F.log_softmax(self.cate(mid2), dim = 1)

class loss(tnn.Module):
    """
    we tried to use mse there with categoryOutput and categoryTarget,
    but the transformation of tensor array is much harder than we expected, 
    so we use nll for category and crossentropy for rate since rate has only two outcome while category has more
    As you can see from the calculation of final loss,
    We believe that appropriate weighting for rateloss and cateloss can provide a good enough loss
    even though we fail to use mseloss from tnn.
    In the end, we find that ratio as 0.245 for rating and 0.755 for categpry yields best result
    """

    def __init__(self):
        super(loss, self).__init__()
        self.categoryloss = None
        self.ratingloss = None

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        self.categoryloss = F.nll_loss(categoryOutput,categoryTarget)
        self.ratingloss = F.cross_entropy(ratingOutput,ratingTarget)
        return self
    
    def backward(self):
        self.ratingloss.backward(retain_graph=True)
        self.categoryloss.backward()
        pass

    def item(self):
        Rweight = 0.245
        return self.ratingloss.item()*Rweight + self.categoryloss.item()*(1-Rweight)

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

"""
the choice of training options is explained in the answer
"""

trainValSplit = 0.99
batchSize = 29
epochs = 38
optimiser = toptim.SGD(net.parameters(), lr=0.008)
    