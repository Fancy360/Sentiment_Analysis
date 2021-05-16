on


# 1. Library imports
import uvicorn
from fastapi import FastAPI
import fasttext

# 2. Create the app object
app = FastAPI()

classifier=fasttext.load_model("model_filename.vec")

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello world.'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to Sentiment analysis using fasttext': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted label
@app.post('/predict')
def predict_text(data: str):
 
    sentence=data
 
    prediction = classifier.predict(sentence)
    if(prediction[0][0]=='__label__2'):
        prediction="Positive sentiment"
    else:
        prediction="Negative sentiment"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
# uvicorn app:app --reload