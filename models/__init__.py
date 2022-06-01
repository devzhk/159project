from .treba import TREBA_model


model_dict = {
    'treba_model' : TREBA_model
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError