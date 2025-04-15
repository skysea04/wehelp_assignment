import torch
from title_classification_model_trainer import TitleClassifier

model_path = "data/title_classifier_30_1000_800_9_1000000.pth"
classifier: TitleClassifier = torch.load(model_path)

print(classifier)
