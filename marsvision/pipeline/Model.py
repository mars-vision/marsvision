
import sklearn
class Model:
    Model.PYTORCH = "pytorch"
    Model.SKLEARN = "sklearn"

    def __init__(self, model_type: str = MODEL.PYTORCH):
        self.model_type = model_type
    def train_model():
        if model_type == MODEL.PYTORCH:
            self.model = train_pytorch_cnn()
        if model_type == MODEL.SKLEARN
            self.model = train_sklearn()

    def train_pytorch_cnn():
        # pytorch code here    

        return model

    def train_sklearn():
        # sklearn code here

        return model

    def save_model():
        # save model to file

    def load_model();
        #load model from file

    

    
