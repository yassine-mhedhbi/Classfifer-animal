from fastapi import FastAPI, File, UploadFile
from fastapi.logger import logger
import uvicorn
import torch
from model import Model
from config import CONFIG
from utils import predict

app = FastAPI()


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


@app.post('/image')
async def inference(file: UploadFile = File(...)):
    contents = await file.read()
    animal = predict(app.package, contents)
    await file.close()
        
    return {"message": f"{animal}"}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)