import json 
import string
import numpy as np
from tensorflow import keras

from nltk.stem import WordNetLemmatizer 
import nltk
import colorama 
colorama.init()
from colorama import Fore, Style, Back
from ftlangdetect import detect
from googletrans import Translator
from translate import Translator
nltk.download("punkt")
nltk.download("wordnet")
import random


lemmatizer = WordNetLemmatizer()

with open("datasets/intents.json") as file:
    data = json.load(file)
    
    
with open("datasets/intentsfr.json") as file:
    datafr = json.load(file)    
    
    
model = keras.models.load_model('my_models/chat_model1')
modelfr = keras.models.load_model('my_models/chat_modelfr1')



words = [] #For Bow model/ vocabulary for patterns
classes = [] #For Bow  model/ vocabulary for tags
data_X = [] #For storing each pattern
data_y = [] #For storing tag corresponding to each pattern in data_X 
# Iterating over all the intents


wordsfr = [] #For Bow model/ vocabulary for patterns
classesfr = [] #For Bow  model/ vocabulary for tags
datafr_X = [] #For storing each pattern
datafr_y = [] #For storing tag corresponding to each pattern in data_X 
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



for intent in datafr["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern) # tokenize each pattern 
        wordsfr.extend(tokens) #and append tokens to words
        datafr_X.append(pattern) #appending pattern to data_X
        datafr_y.append(intent["tag"]) ,# appending the associated tag to each pattern 
    
    # adding the tag to the classes if it's not there already 
    if intent["tag"] not in classesfr:
        classesfr.append(intent["tag"])


# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))


# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
wordsfr = [lemmatizer.lemmatize(word.lower()) for word in wordsfr if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
wordsfr = sorted(set(wordsfr))
classesfr = sorted(set(classesfr))


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


## For data in french

def clean_textfr(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_wordsfr(text, vocab): 
  tokens = clean_textfr(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_classfr(text, vocab, labels): 
  bow = bag_of_wordsfr(text, vocab)
  result = modelfr.predict(np.array([bow]))[0] #Extracting probabilities
  thresh = 0.5
  y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True) #Sorting by values of probability in decreasing order
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]]) #Contains labels(tags) for highest probability 
  return return_list

def get_responsefr(intents_list, intents_json): 
  if len(intents_list) == 0:
    result = "Désolé! Je ne comprends pas."
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


def detect_language(text):
  result = detect(text)
  return result['lang']

def translate_text(text, target_language):
  translator = Translator(to_lang=target_language)
  translation = translator.translate(text)
  return translation

def chatbot(message):
    user_language = detect_language(message)
    print("User Language:", user_language)  # Vérifier la langue détectée

    if user_language == "fr":
      intents = pred_classfr(message, wordsfr, classesfr)
      result = get_responsefr(intents, datafr)

    elif user_language =="en" or message == "Hello":
      intents = pred_class(message, words, classes)
      result = get_response(intents, data)
      
      
    else:
      translated_message = translate_text(message, 'en')  # Traduire le message de l'utilisateur en anglais
      
      # Utiliser votre modèle de chatbot entraîné en anglais pour générer une réponse en anglais
      intents = pred_class(translated_message, words, classes)
      print("English Intents:", intents)  # Vérifier les classes prédites en anglais

      result = get_response(intents, data)
      print("English Result:", result)  # Vérifier la réponse générée en anglais

      
      result = translate_text(result, user_language)  # Traduire la réponse en langage de l'utilisateur
      print("Translated Result:", result)  # Vérifier la réponse traduite
      
    return result
  
chat()