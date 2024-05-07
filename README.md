# Bao_Chatbot

## Presentation of the project

Bao is a multilingual chatbot created from scratch (without the use of a pre-trained model) specialising in the field of agriculture. Its aim is to help farmers find out more about the right conditions for a specific crop to grow properly. Although it is specialised in this field, it can also answer open questions or conversational questions.


## Version constraints
- Python*: Approximately 3.7 to 3.12


## Bao_chatbot project structure
```
Bao_chatbot/
│
├── datasets/
│ ├── intents.json
│ ├── intentsfr.json
│ ├── links.txt
|
│── bao_env/
|
|
├── my_app_Flask/
│ ├── templates/
│ │ └── chat.html
│ ├── app.py
│ │   
│ ├── chat.py   
│
├── my_models/
│   
│
├── requirements.txt
│
└── train.ipynb
```


## Explanation of the structure :

- datasets/ : Contains mainly the datasets used to train the models.

    - intents.json : Contains mainly data in English.

    - intentsfr.json : Contains mainly French-language data.

    - links.txt : Contains the links I used to create this dataset.

- bao_env/ : is the virtual environment we created to install all the dependencies needed to launch the project.

- my_app_Flask/ : This directory contains the chat interface, as well as the api and the flask api that we've integrated into our interface. :

    - templates/ : Contains the single chat.html file, which is our chat interface.
    - app.py/ : This is our flask api.
    - chat.py : Contains the useful functions for interacting with the chat.
    
- my_models : Contains the models trained in the train.ipynb file.

- requirements.txt : A file listing the project's dependencies. It is used to install all the dependencies in our virtual environment.


- train.ipynb: file containing the training and evaluation of the model.


## Procedure for launching Bao_chatbot

To start you need to create a virtual environment in the project directory using the following:

- On Linux:

```console
python -m venv bao_env
```

- On Windows:

```console
py -m venv bao_env
```

Then you need to activate the virtual environment using the following command:

- On Linux:

```console
source bao_env/bin/activate
```

- On Windows:

```console
bao_env\Scripts\activate
```

Then install all the python libraries with the specified version using the command :

```console
pip install -r requirements.txt
```

Finally to launch the application, go into the ```my_app_Flask``` folder and run the ```app.py``` file.

N.B : We can also test the operation of the non-multilingual version of the bot, trained only on English data on the terminal, by executing the file ```chat.py``` located in the ```my_app_Flask``` folder.

### Warning !! 
The fasttext-langdetect library fasttext-langdetect can emit errors during the running of app.py file because it uses network connection.
