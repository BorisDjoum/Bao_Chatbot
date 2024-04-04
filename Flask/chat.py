import json 
import string
import numpy as np
from tensorflow import keras

from nltk.stem import WordNetLemmatizer 
import nltk
import colorama 
colorama.init()
from colorama import Fore, Style, Back
from PyQt6.QtCore import QTimer

import random


lemmatizer = WordNetLemmatizer()

with open("datasets/intents.json") as file:
    data = json.load(file)
    
model = keras.models.load_model('chat_model')

words = [] #For Bow model/ vocabulary for patterns
classes = [] #For Bow  model/ vocabulary for tags
data_X = [] #For storing each pattern
data_y = [] #For storing tag corresponding to each pattern in data_X 
# Iterating over all the intents

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern) # tokenize each pattern 
        words.extend(tokens) #and append tokens to words
        data_X.append(pattern) #appending pattern to data_X
        data_y.append(intent["tag"]) ,# appending the associated tag to each pattern 
    
    # adding the tag to the classes if it's not there already 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))

#7 Preprocessing the Input

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0] #Extracting probabilities
  thresh = 0.5
  y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True) #Sorting by values of probability in decreasing order
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]]) #Contains labels(tags) for highest probability 
  return return_list

def get_response(intents_list, intents_json): 
  if len(intents_list) == 0:
    result = "Sorry! I don't understand."
  else:
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents: 
      if i["tag"] == tag:
        result = random.choice(i["responses"])
        break
    return result

def chat():
   
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        intents = pred_class(inp, words, classes)
        result = get_response(intents, data)
        print(Fore.GREEN + "Bao:" + Style.RESET_ALL , result)

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

#print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
#chat()

#def display_response_progressively(self, solution):
#  self.current_index = 0
#  self.text_to_display = solution
#
#  self.timer = QTimer()
#  self.timer.timeout.connect(self.update_response_label)
#  self.timer.start(25)  # Durée entre chaque caractère (en millisecondes)


def chatbot(message):
  
  intents = pred_class(message, words, classes)
  result = get_response(intents, data)
  return result
