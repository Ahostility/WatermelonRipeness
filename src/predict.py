import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .dirs import DIR_DATA_MODELS
from .process_data import extract_features_mfcc
from .check_data import is_watermelon

def predict(path):
    predict_classes = [
        'unripe',
        'ripe',
        'overripe'
    ]

    if not is_watermelon(path):
        return 'nowatermelon'

    model = models.resnet50()
    model.conv1 = nn.Sequential(
        nn.Conv2d(1, 3, 3, 1, 1, 1, 1),
        model.conv1
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.fc.in_features, len(predict_classes))
    )

    model_path = DIR_DATA_MODELS / 'model.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    features = np.zeros((1, 1, 1, 64))
    features[0, 0, 0] = extract_features_mfcc(path, n_mfcc=64)

    pred = model(torch.FloatTensor(features)).detach()
    prob = F.softmax(pred, dim=1)

    return predict_classes[prob.argmax()]