import nltk
#from nltk.lemmatize.lancaster import Lancasterlemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import speech_recognition as sr

r = sr.Recognizer()

from gtts import gTTS
import os




with open("intents.json") as file:
    data = json.load(file)
ch=1
if ch!=1:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
else:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])],name='chat_in')
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax",name='chat_out')
net = tflearn.regression(net)

model = tflearn.DNN(net)

#if ch!=1:
try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    
# Remove train ops
with net.graph.as_default():
    del tensorflow.get_collection_ref(tensorflow.GraphKeys.TRAIN_OPS)[:]
 
    
model.save("model.tflearn")


with tensorflow.Session() as session:
    my_saver = tensorflow.train.import_meta_graph('model.tflearn.meta')
    my_saver.restore(session, tensorflow.train.latest_checkpoint('.'))
    
    # Rest of the code goes here
    output_node_names = [n.name for n in tensorflow.get_default_graph().as_graph_def().node]
    #print(output_node_names)
    frozen_graph = tensorflow.graph_util.convert_variables_to_constants(
    session,
    session.graph_def,
    ['chat_out/Softmax']
    )
    
    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph.SerializeToString())



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        with sr.Microphone() as source:
            audio = r.listen(source,phrase_time_limit=5)

    
        try:
            speech_to_text=r.recognize_google(audio,language='en-US')
            print("YOU:"+speech_to_text)
        
            #print("You:")
            inp = speech_to_text
            if inp.lower() == "quit":
                break
    
            results = model.predict([bag_of_words(inp, words)])
            results_index = numpy.argmax(results)
            tag = labels[results_index]
    
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
    
            #print(random.choice(responses))
            
            text_to_speech=random.choice(responses)
            print(text_to_speech)
            tts = gTTS(text=text_to_speech, lang='en')
            tts.save("C:/Users/Raghav N G/Downloads/chatbot/intro.mp3")

            os.system("start intro.mp3")
        except:
            pass

chat()