#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

os.system('pip install --quiet transformers==2.9.0')
os.system('pip install --quiet nltk==3.4.5')


# In[14]:


os.system('pip install torch==1.11.0')


# In[19]:


os.system('pip install tabulate')


# In[39]:


os.system('pip install sense2vec==1.0.2')


# In[40]:


if not (os.path.isfile('s2v_reddit_2015_md.tar.gz')):
	os.system('wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz')
	os.system('tar -xvf  s2v_reddit_2015_md.tar.gz')


# In[41]:


# load sense2vec vectors
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('s2v_old')


# In[42]:


from collections import OrderedDict
def sense2vec_get_words(word,s2v):
    output = []
    word = word.lower()
    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)
    # sense ="cricket|NOUN"
    # print("Sense:",sense)
    if(sense!=None):

      # print(dir(s2v.most_similar(sense)))
      most_similar = s2v.most_similar(sense, n=20)

      # print ("most_similar ",most_similar)

      for each_word in most_similar:
          append_word = each_word[0].split("|")[0].replace("_", " ").lower()
          if append_word.lower() != word:
              output.append(append_word.title())

      out = list(OrderedDict.fromkeys(output))
      return out
    else:
      print("\nCould not generate distracters as Sense not found\n")

word = "learning"
distractors = sense2vec_get_words(word,s2v)

print ("Distractors for ",word, " : ")
print(distractors)


# In[5]:


# connect your personal google drive to store the trained model
# from google.colab import drive
# drive.mount('/content/gdrive')


# In[4]:


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
'''
sentence1 = "Srivatsan loves to watch cricket during his free time"
sentence2 = "Srivatsan is annoyed by a cricket in his room"
'''

# In[6]:


import nltk
nltk.download('omw-1.4')


# In[7]:

'''
# An example of a word with two different senses
original_word = "cricket"

syns = wn.synsets(original_word,'n')

for syn in syns:
  print (syn, ": ",syn.definition(),"\n" )

'''
# In[8]:


# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

'''
synset_to_use = wn.synsets(original_word,'n')[0]
distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

print ("\noriginal word: ",original_word.capitalize())
print (distractors_calculated)


original_word = "cricket"
synset_to_use = wn.synsets(original_word,'n')[1]
distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

print ("\noriginal word: ",original_word.capitalize())
print (distractors_calculated)

'''
# In[11]:


import os
import zipfile

bert_wsd_pytorch = "bert_base.zip"
#bert_wsd_pytorch = "My Drive/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6.zip"
# extract_directory = "https://drive.google.com/drivemy-drive"
# extract_directory ="/"
extracted_folder = bert_wsd_pytorch.replace(".zip","")

#  If unzipped folder exists don't unzip again.
if not os.path.isdir(extracted_folder):
  with zipfile.ZipFile(bert_wsd_pytorch, 'r') as zip_ref:
#       zip_ref.extractall(extract_directory)
      zip_ref.extractall()
else:
  print (extracted_folder," is extracted already")


# In[16]:


import torch
import math
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer

class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


# def _forward(args, model, batch):
#     batch = tuple(t.to(args.device) for t in batch)
#     outputs = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])

#     return model.dropout(outputs[1])
    

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = "bert_base.zip (Unzipped Files)/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"


model = BertWSD.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
# add new special token
if '[TGT]' not in tokenizer.additional_special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    assert '[TGT]' in tokenizer.additional_special_tokens
    model.resize_token_embeddings(len(tokenizer))
    
model.to(DEVICE)
model.eval()


# In[17]:


import csv
import os
from collections import namedtuple

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


#from nltk.corpus import  wordnet_ic as wn

import torch
from tqdm import tqdm

GlossSelectionRecord = namedtuple("GlossSelectionRecord", ["guid", "sentence", "sense_keys", "glosses", "targets"])
BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"])



def _create_features_from_records(records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False):
    """ Convert records to list of features. Each feature is a list of sub-features where the first element is
        always the feature created from context-gloss pair while the rest of the elements are features created from
        context-example pairs (if available)
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        tokens_a = tokenizer.tokenize(record.sentence)

        sequences = [(gloss, 1 if i in record.targets else 0) for i, gloss in enumerate(record.glosses)]

        pairs = []
        for seq, label in sequences:
            tokens_b = tokenizer.tokenize(seq)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            pairs.append(
                BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label)
            )

        features.append(pairs)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# In[20]:


import re
import torch
from tabulate import tabulate
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer
import time


MAX_SEQ_LENGTH = 128

def get_sense(sent):
  re_result = re.search(r"\[TGT\](.*)\[TGT\]", sent)
  if re_result is None:
      print("\nIncorrect input format. Please try again.")

  ambiguous_word = re_result.group(1).strip()

  results = dict()

  wn_pos = wn.NOUN
  for i, synset in enumerate(set(wn.synsets(ambiguous_word, pos=wn_pos))):
      results[synset] =  synset.definition()

  if len(results) ==0:
    return (None,None,ambiguous_word)

  # print (results)
  sense_keys=[]
  definitions=[]
  for sense_key, definition in results.items():
      sense_keys.append(sense_key)
      definitions.append(definition)


  record = GlossSelectionRecord("test", sent, sense_keys, definitions, [-1])

  features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_token_segment_id=0,
                                            disable_progress_bar=True)[0]

  with torch.no_grad():
      logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
      # for i, bert_input in tqdm(list(enumerate(features)), desc="Progress"):
      for i, bert_input in list(enumerate(features)):
          logits[i] = model.ranking_linear(
              model.bert(
                  input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
              )[1]
          )
      scores = softmax(logits, dim=0)

      preds = (sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True))


  # print (preds)
  sense = preds[0][0]
  meaning = preds[0][1]
  return (sense,meaning,ambiguous_word)

'''
sentence1 = "Srivatsan loves to watch **cricket** during his free time"


sentence_for_bert = sentence1.replace("**"," [TGT] ")
sentence_for_bert = " ".join(sentence_for_bert.split())
sense,meaning,answer = get_sense(sentence_for_bert)

print (sentence1)
print (sense)
print (meaning)

sentence2 = "Srivatsan is annoyed by a **cricket** in his room"
sentence_for_bert = sentence2.replace("**"," [TGT] ")
sentence_for_bert = " ".join(sentence_for_bert.split())
sense,meaning,answer = get_sense(sentence_for_bert)

print ("\n-------------------------------")
print (sentence2)
print (sense)
print (meaning)

'''
# In[21]:


from transformers import T5ForConditionalGeneration,T5Tokenizer

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

def get_question(sentence,answer):
  text = "context: {} answer: {} </s>".format(sentence,answer)
  print (text)
  max_len = 256
  encoding = question_tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = question_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=200)


  dec = [question_tokenizer.decode(ids) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question

'''
sentence1 = "Srivatsan loves to watch **cricket** during his free time"
sentence2 = "Srivatsan is annoyed by a **cricket** in his room"


answer = "cricket"

sentence_for_T5 = sentence1.replace("**"," ")
sentence_for_T5 = " ".join(sentence_for_T5.split()) 
ques = get_question(sentence_for_T5,answer)
print (ques)


print ("\n**************************************\n")
sentence_for_T5 = sentence2.replace("**"," ")
sentence_for_T5 = " ".join(sentence_for_T5.split()) 
ques = get_question(sentence_for_T5,answer)
print (ques)
'''

# In[22]:


def getMCQs(sent):
  sentence_for_bert = sent.replace("**"," [TGT] ")
  sentence_for_bert = " ".join(sentence_for_bert.split())
  # try:
  sense,meaning,answer = get_sense(sentence_for_bert)
  if sense is not None:
    distractors = get_distractors_wordnet(sense,answer)
  else: 
    distractors = ["Word not found in Wordnet. So unable to extract distractors."]
  sentence_for_T5 = sent.replace("**"," ")
  sentence_for_T5 = " ".join(sentence_for_T5.split()) 
  ques = get_question(sentence_for_T5,answer)
  return ques,answer,distractors,meaning


'''
print ("\n")
question,answer,distractors,meaning = getMCQs(sentence1)
print (question)
print (answer)
print (distractors)
print (meaning)

print ("\n")
question,answer,distractors,meaning = getMCQs(sentence2)
print (question)
print (answer)
print (distractors)
print (meaning)

'''
# In[23]:


# More examples

#sentence = "John went to river **bank** to cry"
#sentence = "John went to deposit money in the **bank**"
# sentence = "John bought a **mouse** for his computer"
# sentence = "John saw a **mouse** under his bed"

'''
print ("\n")
question,answer,distractors,meaning = getMCQs(sentence)
print (question)
print (answer)
print (distractors)
print (meaning)
'''
# In[24]:

'''
# More examples

#sentence = "John went to river **bank** to cry"
sentence = ['John went to deposit **money** in the bank','John went to river **bank** to cry','John bought a **mouse** for his computer.','John saw a **mouse** under his bed.']
# sentence = "John bought a **mouse** for his computer."
# sentence = "John saw a **mouse** under his bed."


print ("\n")
for i in range(0,len(sentence)):
  question,answer,distractors,meaning = getMCQs(sentence[i])
  print (question)
  print (answer)
  print (distractors)
  print ("\n")
  print("Next Question")
  print ("\n")
  # print(meaning)

'''
# In[29]:


# full_text='''A lion was once sleeping in the jungle when a mouse started running up and down his body just for fun. This d.i.sturbed the lion’s sleep, and he woke up quite angry. He was about to eat the mouse when the mouse desperately requested the lion to set him free. 
# “I promise you, I will be of great help to you someday if you save me.”The lion laughed at the mouse’s confidence and let him go.'''


# In[26]:


import nltk
nltk.download('punkt')


# In[30]:


# from nltk.tokenize import sent_tokenize
# print("\nOriginal string:")
# print(full_text)
# token_text = sent_tokenize(full_text)
# print("\nSentence-tokenized copy in a list:")
# print(token_text)
# print("\nRead the list:")
# for s in token_text:
#     print(s)


# In[31]:


# print ("\n")
# for i in range(0,len(token_text)):
#   question,answer,distractors,meaning = getMCQs(token_text[i])
#   print (question)
#   print (answer)
#   print (distractors)
#   print ("\n")
#   print("Next Question")
#   print ("\n")


# In[ ]:


# !pip install pytextrank


# In[ ]:


# import spacy
# import pytextrank
# # example text
# text = """
# Though the term statistical learning is fairly new, many of the concepts that underlie the field were developed long ago. At the beginning of the nineteenth century, Legendre and Gauss published papers on the method of least squares, which implemented the earliest form of what is now known as linear regression. The approach was first successfully applied to problems in astronomy. Linear regression is used for predicting quantitative values, such as an individual’s salary. In order to predict qualitative values, such as whether a patient survives or dies, or whether the stock market increases or decreases, Fisher proposed linear discriminant analysis in 1936. In the 1940s, various authors put forth an alternative approach, logistic regression. In the early 1970s, Nelder and Wedderburn coined the term "generalized linear models" for an entire class of statistical learning methods that include both linear and logistic regression as special cases. By the end of the 1970s, many more techniques for learning from data were available. However, they were almost exclusively linear methods, because fitting non-linear relationships was computationally infeasible at the time. By the 1980s, computing technology had finally improved sufficiently that non-linear methods were no longer computationally prohibitive. In the mid 1980s Breiman, Friedman, Olshen and Stone introduced classification and regression trees, and were among the first to demonstrate the power of a detailed practical implementation of a method, including cross-validation for model selection. Hastie and Tibshirani coined the term "generalized additive models" in 1986 for a class of non-linear extensions to generalized linear models, and also provided a practical software implementation. Since that time, inspired by the advent of machine learning and other disciplines, statistical learning has emerged as a new subfield in statistics, focused on supervised and unsupervised modeling and prediction. In recent years, progress in statistical learning has been marked by the increasing availability of powerful and relatively user-friendly software, such as the popular and freely available R system. This has the potential to continue the transformation of the field from a set of techniques used and developed by statisticians and computer scientists to an essential toolkit for a much broader community.
#  """
# keys=[]
# keys1=[]
# # load a spaCy model, depending on language, scale, etc.
# nlp = spacy.load("en_core_web_sm")
# # add PyTextRank to the spaCy pipeline
# nlp.add_pipe("textrank")
# doc = nlp(text)
# # examine the top-ranked phrases in the document
# for phrase in doc._.phrases[:10]:
#     # print(phrase.text,phrase.rank)
#     keys.append(phrase.text)
#     keys1.append(phrase.rank)
# print(keys)
# print(keys1) 


# In[32]:

import numpy as np
#full_text =
'''
Though the term statistical learning is fairly new, many of the concepts that underlie the field were developed long ago. At the beginning of the nineteenth century, Legendre and Gauss published papers on the method of least squares, which implemented the earliest form of what is now known as linear regression. The approach was first successfully applied to problems in astronomy. Linear regression is used for predicting quantitative values, such as an individual’s salary. In order to predict qualitative values, such as whether a patient survives or dies, or whether the stock market increases or decreases, Fisher proposed linear discriminant analysis in 1936. In the 1940s, various authors put forth an alternative approach, logistic regression. In the early 1970s, Nelder and Wedderburn coined the term "generalized linear models" for an entire class of statistical learning methods that include both linear and logistic regression as special cases. By the end of the 1970s, many more techniques for learning from data were available. However, they were almost exclusively linear methods, because fitting non-linear relationships was computationally infeasible at the time. By the 1980s, computing technology had finally improved sufficiently that non-linear methods were no longer computationally prohibitive. In the mid 1980s Breiman, Friedman, Olshen and Stone introduced classification and regression trees, and were among the first to demonstrate the power of a detailed practical implementation of a method, including cross-validation for model selection. Hastie and Tibshirani coined the term "generalized additive models" in 1986 for a class of non-linear extensions to generalized linear models, and also provided a practical software implementation. Since that time, inspired by the advent of machine learning and other disciplines, statistical learning has emerged as a new subfield in statistics, focused on supervised and unsupervised modeling and prediction. In recent years, progress in statistical learning has been marked by the increasing availability of powerful and relatively user-friendly software, such as the popular and freely available R system. This has the potential to continue the transformation of the field from a set of techniques used and developed by statisticians and computer scientists to an essential toolkit for a much broader community.
'''
full_text=''

with open('passage.txt','r') as f:
	full_text = ''.join(f.readlines())
f.close()

# In[33]:


#answers='non-linear methods,linear methods,linear regression,linear models,statistical learning methods,non-linear extensions,linear discriminant analysis,fitting non-linear relationships,statistical learning,logistic regression'.split(',')

answers = np.load("keywords.npy")


# In[34]:


print(answers)


# In[35]:


# Sequences generation(without overlapping)----------

def get_context(context):
  context = context.split('.')
  # print(context,len(context))
  s = ''
  for i in range(0,len(context)-3,3):
    t = ''
    x = context[i:i+3]
    x[1],x[2] = x[1].strip(),x[2].strip()		
    x[1],x[2] = x[1].replace(x[1][0],x[1][0].lower()),x[2].replace(x[2][0],x[2][0].lower())
    # print(x)
    # break
    
    t+=', '.join(x)+'.'
    # print(t)
    # s+=','.join(x)+'.'
    # print(t)
    f = 0
    for j in answers:
        if j in t:
          # a = j
          f = 1
    if f == 1:
      while 1:
        ch = np.random.choice(answers)
        #print(t)
        # print(ch)
      
        if ch in t:
          t = t.replace(ch,'**'+ch+'**',1)
          # print(t)
          s+=t+''
          # print(s)
          break
      #print(s)
  return s



f=get_context(full_text)
temp=f.replace('\n','')
print((temp))


# In[36]:


temp


# In[37]:


full_text=temp


# In[38]:

from nltk.tokenize import sent_tokenize
print("\nOriginal string:")
print(full_text)
token_text = sent_tokenize(full_text)
print("\nSentence-tokenized copy in a list:")
print(token_text)


with open('ques_sense2vec.txt','w') as f:
	f.close()
 
with open('ques_wordnet.txt','w') as f:
	f.close()

# In[43]:


#Using sense2vec
print ("\n")
print("Using Sense2vec\n")
for i in range(0,len(token_text)):
  question,answer,distractors,meaning = getMCQs(token_text[i])
  print (question)
  print (answer)
  distractors = sense2vec_get_words(answer,s2v)
  print (distractors)
  with open('ques_sense2vec.txt','a') as f:
  	f.writelines('Question:'+question)
  	f.writelines('\nAnswer: '+ answer)
  	f.writelines('\nDistractors: '+','.join(distractors)+'\n\n')
  	f.close()
  print ("\n")
  print("Next Question")
  print ("\n")


# In[44]:


#Using wordnet
print ("\n")
print("Using Wordnet\n")
for i in range(0,len(token_text)):
  question,answer,distractors,meaning = getMCQs(token_text[i])
  print (question)
  print (answer)
  print (distractors)
  with open('ques_wordnet.txt','a') as f:
  	f.writelines('Question:'+question)
  	f.writelines('\nAnswer: '+ answer)
  	f.writelines('\nDistractors: '+','.join(distractors)+'\n\n')
  	f.close()
  print ("\n")
  print("Next Question")
  print ("\n")


# In[ ]:





# In[ ]:




