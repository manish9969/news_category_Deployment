import numpy as np 
from flask import Flask, request, jsonify, render_template 
import pickle

app = Flask(__name__) # intialize app with flask
count_vectorizer = pickle.load( open('CountVectorizer.pickle', 'rb') ) #  rb:read mode
Classifier = pickle.load(open('classifier.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST']) # form action call this route & perfm fn
def predict(): # for rendering results (predctn) on HTML GUI
    news = str(request.form['news'])   # request data from text box of html write code for preprocessig
    
    
    new_data = [news]
    new_vector = count_vectorizer.transform(new_data)
    pred = Classifier.predict(new_vector)
    return render_template('index.html', prediction_text='category of the news should be {}'.format(pred[0].upper()) )
   









if __name__ == '__main__':
    app.run(debug=True)



















