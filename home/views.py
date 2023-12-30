from django.shortcuts import render
from django.http import HttpResponse

# Tool dependencies
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import Model,load_model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import io
import base64

# Create your views here.

# def index(request):

#     # Page from the theme 
#     return render(request, 'pages/index.html')

def index(request):
    return render(request, 'pages/home.html')

def classifier(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get image from post request
        f = request.FILES['image']

        # Pass uploaded image to template
        input_image = base64.b64encode(f.read()).decode("utf-8")

        # Preprocess image
        image = Image.open(f).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        temp = []
        temp.append(image)
        temp = np.array(temp)

        # Load cnn model
        model = load_model('./model/cnn_classificator_model.h5')

        # Get cnn model dense layer
        model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_1').output) 

        # Predict image feature using cnn model
        feat_test = model_feat.predict(temp)

        # Load xgboost model
        xb = XGBClassifier()
        xb.load_model('./model/xgboost_classificator_model.json')

        # Prediction
        y_pred = xb.predict(feat_test)

        if(y_pred[0] > 0.5):
            hasil = True
        else:
            hasil = False

        # Round probability score
        probability_score = feat_test[0][0]
        
        # return HttpResponse(hasil)
        return render(request, 'pages/classifier.html', {'hasil' : hasil, 'input_image' : input_image, 'probabilitas' : probability_score})

    return render(request, 'pages/classifier.html')

def generator(request):
    return render(request, 'pages/generator.html')
