from django.shortcuts import render
from django.http import HttpResponse
import requests
    

# Create your views here.
def index(request):
    return render(request, 'main/index.html')

def about(request):
    return render(request, 'main/about.html')

def button(request):
    return render(request, 'main/home.html')

bot = []
user = []

def output(request):
    import random
    import json
    import pickle
    import numpy as np

    import nltk
    from nltk.stem import WordNetLemmatizer

    from tensorflow import keras
    from keras.models import load_model
 
    import json
        
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intents.json').read())

    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[1] = 1
        return np.array(bag)

    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
            
        return return_list

    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    
    msg = request.POST.get('text')
    # if msg:
    ints = predict_class(msg)
    bot.append(get_response(ints, intents))
    user.append(msg)
    return render(request, 'main/home.html', {'list': list(zip(user, bot))})
    # else:
    #     return HttpResponse("<h1>blank phrase!</h1>")
        

