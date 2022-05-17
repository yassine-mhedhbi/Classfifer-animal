from fastapi import FastAPI, File, UploadFile, responses, Request
from fastapi.logger import logger
from requests import request
import uvicorn
import torch
from model import Model
from config import CONFIG
from utils import predict
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    model = Model()
    model.load_state_dict(torch.load(
        CONFIG['MODEL_PATH'], map_location=torch.device(CONFIG['DEVICE'])))
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "model": model
    }

@app.get('/', response_class=responses.HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})
    
@app.post('/', response_class=responses.HTMLResponse)
async def inference(request: Request, file: UploadFile = File(...)):
    print(file)
    contents = await file.read()
    animal = predict(app.package, contents)
    await file.close()
        
    return templates.TemplateResponse("index.html", {'request': request, "animal": animal}) 


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)