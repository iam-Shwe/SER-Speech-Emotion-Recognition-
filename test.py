import librosa
from csv import writer,DictReader
from datetime import datetime
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
  X, sample_rate = librosa.load(file_name)
  result = np.array([])
  if mfcc:
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      result = np.hstack((result, mfccs))
  if chroma:
      stft = np.abs(librosa.stft(X))
      chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
      result = np.hstack((result, chroma))
  if mel:
      mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
      result = np.hstack((result, mel))
  return result
loaded_model = pickle.load(open('voice.pkle', 'rb'))
def predict(filename):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    temp=extract_feature(f'upload/{filename}')
    temp = np.reshape(temp, (1, -1))
    x_data=loaded_model.predict(temp)
    print(x_data[0])
    with open('report/emotion.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([filename,x_data[0],dt_string])
        f_object.close()
        return x_data[0]

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        if(file.filename.split(".")[-1] in ['wav']):
            with open(f"upload/{file.filename}", 'wb') as f:
                f.write(contents)
                a=predict(file.filename) 
            return {"message": "File successfully saved","data":a}
    except Exception as e :
        print(e)
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
@app.get("/csvfile")
def upload():
    with open('report/emotion.csv', 'r') as file:
        reader = DictReader(
            file, fieldnames=['Audio', 'Emotion', 'Time'])
        data = list(reader)
        return data



