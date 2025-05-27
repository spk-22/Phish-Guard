import torch
import torch.nn as nn
import torch.nn.functional as F

class IoTModel(nn.Module):
    """Neural network model for IoT attack detection (77 features)"""
    def __init__(self, input_dim=77):
        super(IoTModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # Binary classification: benign/attack

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class MalwareModel(nn.Module):
    """Neural network model for Malware detection (55 features)"""
    def __init__(self, input_dim=55):
        super(MalwareModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # Binary classification: benign/malware

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class PhishModel(nn.Module):
    """Neural network model for Phishing detection (16 features)"""
    def __init__(self, input_dim=16):
        super(PhishModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)  # Binary classification: benign/phishing

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DDoSModel(nn.Module):
    """Neural network model for DDoS attack detection (8 features)"""
    def __init__(self, input_dim=8):
        super(DDoSModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)  # Binary classification: benign/ddos

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class FusionClassifier(nn.Module):
    """
    Global fusion model that combines outputs from all local models
    to make final attack/benign classification
    """
    def __init__(self, num_models=4):
        super(FusionClassifier, self).__init__()
        # Input: concatenated logits from 4 models (4 models * 2 classes = 8 features)
        self.fc1 = nn.Linear(num_models * 2, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # Final binary classification: benign/attack

    def forward(self, logits_list):
        """
        Args:
            logits_list: List of logit tensors from [IoT, Malware, Phish, DDoS] models
        
        Returns:
            Global classification logits (batch_size, 2)
        """
        # Concatenate all model outputs along feature dimension
        x = torch.cat(logits_list, dim=1)  # Shape: (batch_size, num_models * 2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        
        return self.fc4(x)  # Return raw logits for final classification