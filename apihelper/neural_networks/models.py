from sentence_transformers import SentenceTransformer
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def load_body_model(model_name):
    model =  SentenceTransformer(model_name)
    logger.info(f"embedding model with nr of parameters: {sum(p.numel() for p in model.parameters())}")
    return model

class FullyConnectedNN(nn.Module):
    def __init__(self,input_size,intermediary_size, num_classes):
        super(FullyConnectedNN, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, intermediary_size)  # Assuming 384 input features
        self.fc2 = nn.Linear(intermediary_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

