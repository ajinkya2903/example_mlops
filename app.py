# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load
import uvicorn
import pathlib


# Load the model
spam_clf = load(open("Models\model.pkl",'rb'))

# Load vectorizer
vectorizer = load(open("new_tfidf.pkl", 'rb'))

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}


@app.post("/predict_sentiment")
def predict_sentiment(text_message):

    polarity = ""

    if(not(text_message)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid text message")

    prediction = spam_clf.predict(vectorizer.transform([text_message]))

    if(prediction[0] == 0):
        polarity = "Positive Sentiment"

    elif(prediction[0] == 1):
        polarity = "Negative Sentiment"
        
    return {
            "text_message": text_message, 
            "sentiment_polarity": polarity
           }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)