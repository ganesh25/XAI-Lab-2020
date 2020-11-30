import shap
import torch
import numpy as np

def xai_shap(model, x_train, x_test):
    # label = torch.tensor([entry[0] for entry in batch])
    # text_str = [vars(entry).get('text') for entry in x_train[:10]]
    # print(type(text_str[0][0]))
    x_train = torch.tensor(x_train)
    x_test = torch.tensor(x_test, dtype=torch.float)
    explainer = shap.DeepExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test)

    # shap.initjs()