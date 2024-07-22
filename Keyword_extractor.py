#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
os.system('pip install pytextrank')
#os.system('pip install spacy==3.3.1')


# In[ ]:


import spacy
import pytextrank
import numpy as np

#os.system("")
# example text
#text =
"""
Though the term statistical learning is fairly new, many of the concepts that underlie the field were developed long ago. At the beginning of the nineteenth century, Legendre and Gauss published papers on the method of least squares, which implemented the earliest form of what is now known as linear regression. The approach was first successfully applied to problems in astronomy. Linear regression is used for predicting quantitative values, such as an individual’s salary. In order to predict qualitative values, such as whether a patient survives or dies, or whether the stock market increases or decreases, Fisher proposed linear discriminant analysis in 1936. In the 1940s, various authors put forth an alternative approach, logistic regression. In the early 1970s, Nelder and Wedderburn coined the term "generalized linear models" for an entire class of statistical learning methods that include both linear and logistic regression as special cases. By the end of the 1970s, many more techniques for learning from data were available. However, they were almost exclusively linear methods, because fitting non-linear relationships was computationally infeasible at the time. By the 1980s, computing technology had finally improved sufficiently that non-linear methods were no longer computationally prohibitive. In the mid 1980s Breiman, Friedman, Olshen and Stone introduced classification and regression trees, and were among the first to demonstrate the power of a detailed practical implementation of a method, including cross-validation for model selection. Hastie and Tibshirani coined the term "generalized additive models" in 1986 for a class of non-linear extensions to generalized linear models, and also provided a practical software implementation. Since that time, inspired by the advent of machine learning and other disciplines, statistical learning has emerged as a new subfield in statistics, focused on supervised and unsupervised modeling and prediction. In recent years, progress in statistical learning has been marked by the increasing availability of powerful and relatively user-friendly software, such as the popular and freely available R system. This has the potential to continue the transformation of the field from a set of techniques used and developed by statisticians and computer scientists to an essential toolkit for a much broader community.
"""
text = input("enter passage: ")
with open('passage.txt','w') as f:
	f.writelines(text)
f.close()

keys=[]
keys1=[]
# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")
doc = nlp(text)
# examine the top-ranked phrases in the document
for phrase in doc._.phrases[:10]:
    # print(phrase.text,phrase.rank)
    keys.append(phrase.text)
keys = np.array(keys)
np.save('keywords.npy',keys)

