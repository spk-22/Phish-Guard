import torch
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from models import IoTModel, MalwareModel, PhishModel, DDoSModel, FusionClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Load all models (local attack-specific models and global fusion model)"""
    iot_model = IoTModel(input_dim=77).to(device)
    malware_model = MalwareModel(input_dim=55).to(device)
    phish_model = PhishModel(input_dim=16).to(device)
    ddos_model = DDoSModel(input_dim=8).to(device)
    fusion_model = FusionClassifier(num_models=4).to(device)

    # Load pre-trained weights - FIXED PATHS
    try:
        iot_model.load_state_dict(torch.load("iot_model.pth", map_location=device), strict=False)
        malware_model.load_state_dict(torch.load("malware_model.pth", map_location=device), strict=False)
        phish_model.load_state_dict(torch.load("phish_model.pth", map_location=device), strict=False)
        ddos_model.load_state_dict(torch.load("ddos_model.pth", map_location=device), strict=False)
        fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location=device), strict=False)
        print("✅ All model weights loaded successfully!")
    except FileNotFoundError as e:
        print(f"Warning: Could not load model weights - {e}")
        print("Models will run with random weights")
    except Exception as e:
        print(f"Error loading models: {e}")

    # Set all models to evaluation mode
    for model in [iot_model, malware_model, phish_model, ddos_model, fusion_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return {
        "IoT": iot_model,
        "Malware": malware_model,
        "Phish": phish_model,
        "DDoS": ddos_model,
        "Global": fusion_model
    }

def preprocess_file(uploaded_file, expected_features):
    """Preprocess uploaded file to match expected feature count for specific model"""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        file_name = uploaded_file.name
        
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            raise ValueError("Unsupported file format! Upload CSV or JSON.")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    # Handle IP address columns if present
    def split_ip(ip_str):
        try:
            parts = str(ip_str).split('.')
            if len(parts) == 4:
                return [int(p) for p in parts]
            else:
                return [0, 0, 0, 0]
        except:
            return [0, 0, 0, 0]

    # Process source IP
    if 'src_ip' in df.columns:
        ip_cols = df['src_ip'].apply(split_ip).tolist()
        ip_df = pd.DataFrame(ip_cols, columns=['src_ip_1', 'src_ip_2', 'src_ip_3', 'src_ip_4'])
        df = pd.concat([df, ip_df], axis=1).drop(columns=['src_ip'])

    # Process destination IP
    if 'dst_ip' in df.columns:
        ip_cols = df['dst_ip'].apply(split_ip).tolist()
        ip_df = pd.DataFrame(ip_cols, columns=['dst_ip_1', 'dst_ip_2', 'dst_ip_3', 'dst_ip_4'])
        df = pd.concat([df, ip_df], axis=1).drop(columns=['dst_ip'])

    # Encode categorical columns
    categorical_columns = ['proto', 'type', 'service', 'flag', 'attack_type', 'category']
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col].astype(str))
            except Exception:
                df[col] = 0

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Remove any label columns that shouldn't be used for prediction
    label_columns = ['label', 'class', 'target', 'attack', 'malicious']
    for col in label_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Adjust column count to match expected features
    num_cols = df.shape[1]
    if num_cols > expected_features:
        # Take first N columns if we have too many
        df = df.iloc[:, :expected_features]
    elif num_cols < expected_features:
        # Pad with zeros if we have too few
        for i in range(expected_features - num_cols):
            df[f'pad_col_{i}'] = 0

    # Convert to tensor
    tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    return tensor

def global_inference(preprocessed_data, models):
    """
    Run global fusion model inference to determine attack/benign classification
    and identify which attack type is most likely
    """
    # Get outputs from all local models
    iot_logits = models["IoT"](preprocessed_data["IoT"])
    malware_logits = models["Malware"](preprocessed_data["Malware"])
    phish_logits = models["Phish"](preprocessed_data["Phish"])
    ddos_logits = models["DDoS"](preprocessed_data["DDoS"])

    # Combine logits for fusion model
    logits_list = [iot_logits, malware_logits, phish_logits, ddos_logits]
    fusion_output = models["Global"](logits_list)
    fusion_probs = F.softmax(fusion_output, dim=1)

    # Global prediction (0 = Benign, 1 = Attack)
    pred_class = fusion_probs.argmax(dim=1)
    
    # Calculate confidence scores for each attack type
    # Use the attack probabilities from each local model to determine reasoning
    model_names = ["IoT", "Malware", "Phish", "DDoS"]
    model_confidences = {}
    
    for i, (name, logits) in enumerate(zip(model_names, logits_list)):
        # Get attack probability (class 1) from each local model
        attack_probs = F.softmax(logits, dim=1)[:, 1]
        model_confidences[name] = attack_probs.mean().item()
    
    # Generate reasoning based on model confidence scores
    reasoning = generate_global_reasoning(model_confidences, pred_class, fusion_probs)
    
    return pred_class, fusion_probs, model_confidences, reasoning

def local_inference(input_tensor, model, model_name):
    """
    Run inference on a specific local attack model for detailed classification
    """
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1)
        
        # Generate reasoning for local model
        reasoning = generate_local_reasoning(model_name, probs, pred_class)
        
        return pred_class, probs, reasoning

def generate_global_reasoning(model_confidences, predictions, fusion_probs):
    """Generate human-readable reasoning for global model decisions"""
    attack_count = (predictions == 1).sum().item()
    total_samples = len(predictions)
    benign_count = total_samples - attack_count
    
    reasoning = f"Global fusion model analyzed {total_samples} samples:\n"
    reasoning += f"• Classified {benign_count} as benign and {attack_count} as attacks\n"
    
    if attack_count > 0:
        # Find the most confident attack type
        max_confidence_type = max(model_confidences, key=model_confidences.get)
        max_confidence_score = model_confidences[max_confidence_type]
        
        reasoning += f"• Attack type analysis:\n"
        for attack_type, confidence in sorted(model_confidences.items(), key=lambda x: x[1], reverse=True):
            reasoning += f"  - {attack_type}: {confidence:.4f} confidence\n"
        
        reasoning += f"• Recommended local model: {max_confidence_type} (highest confidence: {max_confidence_score:.4f})\n"
        reasoning += f"• Global model average confidence: {fusion_probs[:, 1].mean().item():.4f}"
    else:
        reasoning += "• All samples classified as benign - no attack-specific analysis needed"
    
    return reasoning

def generate_local_reasoning(model_name, probs, predictions):
    """Generate human-readable reasoning for local model decisions"""
    attack_count = (predictions == 1).sum().item()
    total_samples = len(predictions)
    benign_count = total_samples - attack_count
    avg_confidence = probs[:, 1].mean().item()
    
    reasoning = f"{model_name} model detailed analysis:\n"
    reasoning += f"• Analyzed {total_samples} samples with average confidence: {avg_confidence:.4f}\n"
    reasoning += f"• Classified {benign_count} as benign/other and {attack_count} as {model_name} attacks\n"
    
    if attack_count > 0:
        # Get confidence statistics for attack predictions
        attack_confidences = probs[predictions == 1, 1]
        if len(attack_confidences) > 0:
            min_conf = attack_confidences.min().item()
            max_conf = attack_confidences.max().item()
            reasoning += f"• Attack prediction confidence range: {min_conf:.4f} to {max_conf:.4f}\n"
    
    # Add model-specific insights
    if model_name == "IoT":
        reasoning += "• Specializes in detecting IoT botnet attacks, scanning behavior, and device compromise"
    elif model_name == "Malware":
        reasoning += "• Specializes in detecting malicious software, trojans, ransomware, and file-based threats"
    elif model_name == "Phish":
        reasoning += "• Specializes in detecting phishing attempts, social engineering, and spoofing attacks"
    elif model_name == "DDoS":
        reasoning += "• Specializes in detecting distributed denial of service attacks and traffic flooding"
    
    return reasoning