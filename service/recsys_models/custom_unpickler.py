import os
import pickle


class CustomUnpickler(pickle.Unpickler):
    """
    Класс для того, чтобы получилось импортировать пикл с
    инстансом обученной UserKnn() модели
    """
    def find_class(self, module, name):
        if name == 'UserKnn':
            from ..recsys_models.userknn import UserKnn
            return UserKnn
        return super().find_class(module, name)


def load(path: str):
    with open(os.path.join(path), 'rb') as f:
        return CustomUnpickler(f).load()
