from .treba_model import TREBA_model
from .tmae import TMAE

model_dict = {
    'treba_model' : TREBA_model,
    'tmae': TMAE
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
