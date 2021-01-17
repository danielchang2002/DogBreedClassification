from fastai.vision import (
    load_learner,
    open_image
)
import torch
class Model():
    def __init__(self, path):
        self.learn = load_learner(path)
        self.classes = self.learn.data.classes

    def predict(self, uploaded_file):
        img = open_image(uploaded_file)
        pred_class, pred_idx, outputs = self.learn.predict(img)
        confidences, outputIdxs = torch.topk(outputs, 3)
        outputs = [self.getClass(idx) for idx in outputIdxs]
        confidences = [self.getPercentage(f, 5) for f in confidences]
        return outputs, confidences
    
    def getClass(self, idx):
        c = self.classes[idx]
        output = ' '.join(c.split('-')[1].split('_'))
        return output
    
    def getPercentage(self, f, prec):
        return "{0:.{prec}%}".format(f, prec=prec)
    