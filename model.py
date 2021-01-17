from fastai.vision import (
    load_learner,
    open_image
)
class Model():
    def __init__(self, path):
        self.learn = load_learner(path)

    def predict(self, uploaded_file):
        img = open_image(uploaded_file)
        pred_class, pred_idx, outputs = self.learn.predict(img)
        output = ' '.join(pred_class.obj.split('-')[1].split('_'))
        return output