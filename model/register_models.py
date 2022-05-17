import neptune.new as neptune
from keys import API_KEY, PROJECT

model = neptune.init_model(
    name="ImageNet Animal Classifier",
    key="MOD", 
    project=PROJECT, 
    api_token=API_KEY
)