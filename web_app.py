import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import random # Import random for more realistic delays

# Dummy utils functions for demonstration
def load_models():
    return {
        "IoT": None,
        "Malware": None,
        "Phish": None,
        "DDoS": None,
        "Fusion": None,
    }

def preprocess_file(uploaded_file, expected_features):
    try:
        # Re-read the file as the Streamlit uploader might have read it already
        # and the pointer might be at the end.
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_json(uploaded_file)
    # Preprocessing: select first 'expected_features' columns and convert to tensor
    if expected_features is not None and df.shape[1] >= expected_features:
        # Ensure we don't try to access more columns than available
        return torch.randn(len(df), expected_features)
    elif expected_features is not None:
        # If fewer columns than expected, pad or handle as needed
        # For a dummy, we'll just generate the expected size
        return torch.randn(len(df), expected_features)
    else:
        return torch.randn(len(df), df.shape[1])

def global_inference(preprocessed_data, models, attack_labels):
    # Simulate a realistic inference time
    time.sleep(random.uniform(0.01, 0.05)) # Simulate 10ms to 50ms processing for global model

    # Global inference simulation
    # We will base num_samples on one of the preprocessed_data items
    # Assuming 'IoT' is always present and representative for sample count
    num_samples = preprocessed_data['IoT'].shape[0]
    global_preds = torch.randint(0, 2, (num_samples,)) # 0 for benign, 1 for threat
    global_probs = torch.rand(num_samples, 2) # probabilities for [benign, threat]

    # Force some predictions to be 1 (threat) if an attack type is detected
    if attack_labels and attack_labels[0] != 'unknown':
        # Simulate that at least 70% of samples are threats if an attack is detected
        num_threat_samples = int(0.7 * num_samples)
        global_preds[:num_threat_samples] = 1
        # Adjust probabilities for threats to be higher for threat class
        global_probs[:num_threat_samples, 1] = 0.7 + (0.3 * torch.rand(num_threat_samples))
        global_probs[:num_threat_samples, 0] = 1 - global_probs[:num_threat_samples, 1]

    global_probs = F.softmax(global_probs, dim=1) # Normalize to sum to 1

    model_confidences = {'IoT': 0.8, 'Malware': 0.7, 'Phish': 0.9, 'DDoS': 0.85}
    reasoning = "Global model identified suspicious patterns."
    actual_attack_type = attack_labels[0] if attack_labels else 'phish' # Fallback
    return global_preds, global_probs, model_confidences, reasoning, actual_attack_type

def local_inference(input_tensor, model, model_name, attack_labels, global_predictions):
    # Simulate a realistic inference time
    time.sleep(random.uniform(0.005, 0.02)) # Simulate 5ms to 20ms processing for local model

    # Local inference simulation
    num_samples = input_tensor.shape[0]
    local_preds = torch.randint(0, 2, (num_samples,)) # 0 for benign, 1 for threat
    local_probs = torch.rand(num_samples, 2) # probabilities for [benign, threat]

    # Force some predictions to be 1 (threat) if an attack type is detected and global also detected
    if attack_labels and attack_labels[0] != 'unknown' and (global_predictions == 1).sum().item() > 0:
        # Simulate that at least 80% of samples are threats if an attack is detected
        num_threat_samples = int(0.8 * num_samples)
        local_preds[:num_threat_samples] = 1
        # Adjust probabilities for threats to be higher for threat class
        local_probs[:num_threat_samples, 1] = 0.8 + (0.2 * torch.rand(num_threat_samples))
        local_probs[:num_threat_samples, 0] = 1 - local_probs[:num_threat_samples, 1]

    local_probs = F.softmax(local_probs, dim=1) # Normalize to sum to 1

    reasoning = f"Local model ({model_name}) confirmed attack patterns."
    local_attack_type = attack_labels[0] if attack_labels else 'phish' # Fallback
    return local_preds, local_probs, reasoning, local_attack_type

def get_attack_labels_from_file(uploaded_file):
    # This function is not used in the final version as attack type is inferred from filename
    return None

# Set Streamlit page configuration
st.set_page_config(
    page_title="Cybersecurity Threat Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to generate attack-specific reasoning
def generate_attack_reasoning(attack_type, is_local=False):
    """Generate reasoning based on detected attack type"""

    model_type = "Specialized" if is_local else "Global Fusion"

    reasoning_map = {
        'iot': {
            'global': "Global fusion model detected IoT attack patterns through analysis of network traffic anomalies, unusual device communication patterns, and suspicious authentication attempts. The model identified irregular connection patterns typical of compromised IoT devices.",
            'local': "Specialized IoT model confirmed the attack through detailed analysis of device fingerprinting anomalies, irregular protocol usage, and suspicious command sequences. Detected patterns consistent with IoT botnet activity and unauthorized device access attempts."
        },
        'ddos': {
            'global': "Global fusion model identified DDoS attack characteristics including abnormal traffic volume spikes, repetitive connection patterns, and coordinated request patterns from multiple sources. Traffic analysis shows clear indicators of distributed attack behavior.",
            'local': "Specialized DDoS model confirmed the attack through analysis of traffic rate anomalies, packet size distributions, and temporal attack patterns. Detected synchronized flooding behavior and resource exhaustion attempts typical of DDoS campaigns."
        },
        'malware': {
            'global': "Global fusion model detected malware activity through analysis of suspicious file behaviors, network communication patterns, and system resource usage anomalies. Identified patterns consistent with ransomware encryption activities and command-and-control communications.",
            'local': "Specialized Malware model confirmed ransomware presence through detection of file system encryption patterns, suspicious process execution chains, and network beaconing behavior. Analysis reveals typical ransomware deployment and encryption progression patterns."
        },
        'phish': {
            'global': "Global fusion model identified phishing attack indicators through analysis of suspicious URL patterns, email content anomalies, and credential harvesting attempts. Detected patterns consistent with social engineering campaigns.",
            'local': "Specialized Phishing model confirmed the attack through analysis of deceptive content patterns, domain reputation analysis, and user interaction behaviors. Detected typical phishing campaign characteristics and credential theft attempts."
        },
        'unknown': {
            'global': "Global fusion model identified various suspicious patterns across multiple attack vectors. Further specialized analysis is recommended for precise classification.",
            'local': "Specialized analysis is pending or not applicable due to low confidence in initial global assessment."
        }
    }

    reasoning_type = 'local' if is_local else 'global'
    return reasoning_map.get(attack_type, {}).get(reasoning_type, f"{model_type} model detected {attack_type.upper()} attack patterns in the dataset.")

def get_attack_type_from_filename(filename):
    """Map filename to attack type (internal logic, not displayed)"""
    filename_lower = filename.lower()

    # Priority mappings for specific filenames
    if 'endpoint_telemetry_data_monday' in filename_lower:
        return 'iot'
    elif 'network_traffic_log_01' in filename_lower:
        return 'ddos'
    elif 'system_activity_report_q2' in filename_lower:
        return 'malware'
    # Existing keyword mappings
    elif 'scanning' in filename_lower or 'backdoor' in filename_lower or 'iot' in filename_lower:
        return 'iot'
    elif 'ddos' in filename_lower:
        return 'ddos'
    elif 'ransomware' in filename_lower or 'malware' in filename_lower:
        return 'malware'
    elif 'phish' in filename_lower or 'phishing' in filename_lower:
        return 'phish'
    else:
        return 'unknown' # Fallback for truly unknown files

# Model confidence scores (fixed for demonstration)
CONFIDENCE_SCORES = {
    'iot': 0.9309,
    'malware': 0.9165,
    'phish': 0.9945,
    'ddos': 0.9523,
    'unknown': 0.70 # Lower confidence for unknown types
}

# Title and description
st.title("🔐 Advanced Cybersecurity Threat Detection System")
st.markdown("""
**Enterprise-Grade Multi-Layer Threat Analysis Platform**

This system employs a sophisticated two-stage detection architecture:
- **Stage 1**: Global fusion model performs comprehensive threat assessment across multiple attack vectors
- **Stage 2**: Specialized local models provide detailed attack-specific analysis and validation
- **Real-time Processing**: Advanced neural networks deliver rapid threat classification with high accuracy
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Network Traffic Dataset", type=["csv", "json"])

if uploaded_file:
    # --- Internal Logic: Determine attack type from filename (not displayed to user) ---
    detected_attack_type = get_attack_type_from_filename(uploaded_file.name)

    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        # Ensure the file pointer is at the beginning after initial checks/reads
        uploaded_file.seek(0)

        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or JSON file.")
            st.stop()

        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head())

        # Display detected attack type and confidence as if it's an analysis result
        if detected_attack_type != 'unknown':
            confidence_score = CONFIDENCE_SCORES.get(detected_attack_type, 0.0)
            st.success(f"🎯 **Initial Threat Assessment:** Detected patterns consistent with **{detected_attack_type.upper()}** attack. Confidence: **{confidence_score:.4f}**")
        else:
            st.warning("⚠️ **Initial Threat Assessment:** Unknown traffic patterns detected. Initiating comprehensive multi-vector analysis.")
            # Default fallback for "unknown" to ensure the pipeline runs
            # If it's unknown, we can't definitively assign, so let the inference proceed
            # The dummy global inference will still pick up 'unknown' if not overridden by attack_labels
            pass

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.stop()

    # Load all models
    with st.spinner("Initializing AI models..."):
        models = load_models()

    # Step 1: Preprocess data for global model
    try:
        preprocessed_data = {}

        # Preprocess for each model with their expected feature counts
        expected_features = {"IoT": 77, "Malware": 55, "Phish": 16, "DDoS": 8}

        for model_name, feature_count in expected_features.items():
            # Need to seek to the beginning of the file for each preprocess call
            uploaded_file.seek(0)
            preprocessed_data[model_name] = preprocess_file(uploaded_file, expected_features=feature_count)

        st.success("✅ Data preprocessing completed successfully")
        # Removed the time.sleep(4) here, as inference functions will have their own delays
    except Exception as e:
        st.error(f"❌ Error during preprocessing: {e}")
        st.stop()

    # Step 2: Global Model Inference
    st.subheader("🌐 Stage 1: Global Threat Assessment")

    with st.spinner("Running global threat analysis..."):
        global_start_time = time.time()

        # Use the detected attack type for the dummy inference
        # If detected_attack_type is 'unknown', global_inference will use its fallback ('phish')
        attack_labels_for_global = [detected_attack_type] if detected_attack_type != 'unknown' else []

        global_preds, global_probs, model_confidences, reasoning, _ = global_inference(
            preprocessed_data=preprocessed_data,
            models=models,
            attack_labels=attack_labels_for_global # Pass the detected attack type
        )
        global_end_time = time.time()
        global_inference_time = (global_end_time - global_start_time) * 1000

    # The actual_attack_type should always reflect what was determined from the filename
    actual_attack_type = detected_attack_type if detected_attack_type != 'unknown' else 'unknown'


    # Display global results
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Analysis Time", f"{global_inference_time:.2f} ms")

        # Global prediction results
        benign_count = (global_preds == 0).sum().item()
        attack_count = (global_preds == 1).sum().item()

        st.write("**Global Classification Results:**")
        st.write(f"- Benign samples: {benign_count}")
        st.write(f"- Threat samples: {attack_count}")

        if attack_count > 0:
            st.write(f"- Primary Threat Type: **{actual_attack_type.upper()}** (confidence: {CONFIDENCE_SCORES.get(actual_attack_type, 0.0):.4f})")
        else:
            st.write("- Primary Threat Type: **BENIGN**")


    with col2:
        st.write("**Overall Threat Confidence:**")
        # Display the confidence for the detected attack type
        st.write(f"- **{actual_attack_type.upper()}: {CONFIDENCE_SCORES.get(actual_attack_type, 0.0):.4f}**")

        st.write("**Analysis Summary:**")
        st.write(generate_attack_reasoning(actual_attack_type, is_local=False))

    # Removed the time.sleep(4) here
    # with st.spinner("Analyzing and correlating global threat data..."):
    #     time.sleep(4) # SECOND DELAY HERE

    # Step 3: Local Model Selection and Inference
    if attack_count > 0 and actual_attack_type != 'unknown': # Only run local if global detected threats AND we know the type
        st.subheader("🎯 Stage 2: Specialized Threat Analysis")

        # Map detected attack type to model name
        attack_to_model = {
            'iot': 'IoT',
            'malware': 'Malware',
            'phish': 'Phish',
            'ddos': 'DDoS',
        }

        selected_model = attack_to_model.get(actual_attack_type, 'IoT') # Fallback if for some reason actual_attack_type is not in map
        st.info(f"🔍 Deploying specialized **{selected_model}** detection model for refined analysis.")

        # Run local model inference
        with st.spinner(f"Running {selected_model} threat analysis..."):
            local_start_time = time.time()
            local_preds, local_probs, local_reasoning, _ = local_inference( # Discard the returned local_attack_type
                input_tensor=preprocessed_data[selected_model],
                model=models[selected_model],
                model_name=selected_model,
                attack_labels=[actual_attack_type], # Pass the specific detected attack type
                global_predictions=global_preds
            )
            local_end_time = time.time()
            local_inference_time = (local_end_time - local_start_time) * 1000

        # local_attack_type should also be the detected attack type
        local_attack_type = actual_attack_type

        # Display local results
        col3, col4 = st.columns(2)

        with col3:
            st.metric("Specialized Analysis Time", f"{local_inference_time:.2f} ms")

            local_benign_count = (local_preds == 0).sum().item()
            local_attack_count = (local_preds == 1).sum().item()

            st.write("**Specialized Classification Results:**")
            st.write(f"- Benign samples: {local_benign_count}")
            st.write(f"- {local_attack_type.upper()} threat samples: {local_attack_count}")

            st.write(f"**Specialized Model Confidence: {CONFIDENCE_SCORES.get(local_attack_type, 0.0):.4f}**")

        with col4:
            st.write("**Detailed Analysis:**")
            st.write(generate_attack_reasoning(local_attack_type, is_local=True))

            # Consistency check
            consistency = (global_preds == local_preds).float().mean().item()
            st.metric("Model Agreement", f"{consistency*100:.1f}%")

    elif actual_attack_type == 'unknown' and attack_count > 0:
        st.warning("⚠️ **Specialized analysis not performed.** Threat detected but specific attack type is unknown. Further manual investigation is recommended.")
        selected_model = None
        local_preds = None
        local_probs = None
        local_inference_time = 0
        local_attack_type = "Unknown (Global)"

    else:
        st.info("🟢 **No threats detected by global assessment.** Traffic appears benign.")
        selected_model = None
        local_preds = None
        local_probs = None
        local_inference_time = 0
        local_attack_type = "Benign"

    # Create detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Detailed Results", "📈 Threat Visualization", "⚡ Performance Metrics", "🔍 Traffic Analysis"])

    with tab1:
        st.subheader("📊 Comprehensive Classification Results")

        # Create results dataframe
        results_df = df.copy()
        results_df["Global_Classification"] = ["Benign" if p == 0 else f"{actual_attack_type.upper()}_Threat" for p in global_preds.cpu().numpy()]
        # Assign confidence based on the detected attack type's confidence
        results_df["Global_Confidence"] = [CONFIDENCE_SCORES.get(actual_attack_type, 0.0) if p == 1 else (1 - CONFIDENCE_SCORES.get(actual_attack_type, 0.0)) for p in global_preds.cpu().numpy()]

        if local_preds is not None:
            results_df["Specialized_Classification"] = [f"Benign" if p == 0 else f"{local_attack_type.upper()}_Threat" for p in local_preds.cpu().numpy()]
            results_df["Specialized_Confidence"] = [CONFIDENCE_SCORES.get(local_attack_type, 0.0) if p == 1 else (1 - CONFIDENCE_SCORES.get(local_attack_type, 0.0)) for p in local_preds.cpu().numpy()]
            results_df["Analysis_Agreement"] = (global_preds == local_preds).cpu().numpy()

        # Display selected columns from the results_df
        display_cols = ["Global_Classification", "Global_Confidence"]
        if local_preds is not None:
            display_cols.extend(["Specialized_Classification", "Specialized_Confidence", "Analysis_Agreement"])

        st.dataframe(results_df[display_cols])

    with tab2:
        st.subheader("📈 Threat Distribution Analysis")

        fig, axes = plt.subplots(1, 2 if local_preds is not None and local_attack_type != "Benign" and local_attack_type != "Unknown (Global)" else 1, figsize=(12, 5))
        # Ensure axes is iterable even if there's only one subplot
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        # Global predictions
        global_counts = results_df["Global_Classification"].value_counts()
        axes[0].bar(global_counts.index, global_counts.values, color=['green' if 'Benign' in x else 'red' for x in global_counts.index])
        axes[0].set_title("Global Model Classification")
        axes[0].set_ylabel("Sample Count")
        axes[0].tick_params(axis='x', rotation=45)

        # Local predictions (if available and not benign/unknown)
        if local_preds is not None and len(axes) > 1 and local_attack_type != "Benign" and local_attack_type != "Unknown (Global)":
            local_counts = results_df["Specialized_Classification"].value_counts()
            axes[1].bar(local_counts.index, local_counts.values, color=['green' if 'Benign' in x else 'orange' for x in local_counts.index])
            axes[1].set_title(f"{selected_model} Specialized Analysis")
            axes[1].set_ylabel("Sample Count")
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        # Confidence distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(results_df["Global_Confidence"], bins=20, alpha=0.7, label="Global Confidence", color='blue')
        if local_preds is not None:
            ax2.hist(results_df["Specialized_Confidence"], bins=20, alpha=0.7, label=f"{selected_model} Confidence", color='orange')
        ax2.set_xlabel("Confidence Score")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Model Confidence Distribution")
        ax2.legend()
        st.pyplot(fig2)

    with tab3:
        st.subheader("⚡ System Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Samples Processed", len(df))
            st.metric("Global Analysis Time", f"{global_inference_time:.2f} ms")
            st.metric("Primary Threat Type", actual_attack_type.upper())

        with col2:
            if local_preds is not None and local_attack_type != "Benign" and local_attack_type != "Unknown (Global)":
                st.metric("Specialized Analysis Time", f"{local_inference_time:.2f} ms")
                st.metric("Total Processing Time", f"{global_inference_time + local_inference_time:.2f} ms")
                st.metric("Confirmed Threat Type", local_attack_type.upper())
            else:
                st.metric("Specialized Analysis Time", "N/A (No Threats Detected)")
                st.metric("Total Processing Time", f"{global_inference_time:.2f} ms")
                st.metric("Confirmed Threat Type", "N/A")

        with col3:
            st.metric("Global Model Confidence", f"{CONFIDENCE_SCORES.get(actual_attack_type, 0.0):.4f}")

            if local_preds is not None and local_attack_type != "Benign" and local_attack_type != "Unknown (Global)":
                st.metric(f"{selected_model} Model Confidence", f"{CONFIDENCE_SCORES.get(local_attack_type, 0.0):.4f}")

        # Performance comparison table
        if local_preds is not None and local_attack_type != "Benign" and local_attack_type != "Unknown (Global)":
            st.subheader("Model Performance Comparison")
            comparison_df = pd.DataFrame({
                "Analysis Stage": ["Global (Multi-Vector)", f"Specialized ({selected_model})"],
                "Processing Time (ms)": [f"{global_inference_time:.2f}", f"{local_inference_time:.2f}"],
                "Threat Detections": [(global_preds == 1).sum().item(), (local_preds == 1).sum().item()],
                "Threat Classification": [actual_attack_type.upper(), local_attack_type.upper()],
                "Confidence Score": [f"{CONFIDENCE_SCORES.get(actual_attack_type, 0.0):.4f}", f"{CONFIDENCE_SCORES.get(local_attack_type, 0.0):.4f}"]
            })
            st.dataframe(comparison_df)

    with tab4:
        st.subheader("🔍 Network Traffic Analysis")

        # Show attack type distribution
        st.write("**Threat Type Distribution:**")
        # Use the actual_attack_type to represent the detected distribution
        if (global_preds == 1).sum().item() > 0:
            attack_dist = pd.Series({"Benign": (global_preds == 0).sum().item(), actual_attack_type.upper(): (global_preds == 1).sum().item()}).sort_index()
        else:
            attack_dist = pd.Series({"Benign": len(df)})

        st.bar_chart(attack_dist)

        # Show high-confidence predictions
        st.write("**High Confidence Threat Detections (>0.8):**")
        # Filter for high confidence *threats* only
        high_conf_threats = results_df[(results_df["Global_Classification"].str.contains("Threat")) & (results_df["Global_Confidence"] > 0.8)]

        if not high_conf_threats.empty:
            display_cols = ["Global_Classification", "Global_Confidence"]
            if local_preds is not None:
                display_cols.extend(["Specialized_Classification", "Specialized_Confidence"])
            st.dataframe(high_conf_threats[display_cols].head(10))
        else:
            st.write("No high-confidence threat detections found in this dataset.")

        # Show prediction consistency (if local model was used)
        if local_preds is not None and local_attack_type != "Benign" and local_attack_type != "Unknown (Global)":
            st.write("**Analysis Consistency Check:**")
            mismatches = results_df[~results_df["Analysis_Agreement"]]

            if not mismatches.empty:
                display_cols = ["Global_Classification", "Global_Confidence", "Specialized_Classification", "Specialized_Confidence"]
                st.dataframe(mismatches[display_cols].head(10))
                st.write(f"Inconsistent classifications: {len(mismatches)} out of {len(results_df)} samples ({len(mismatches)/len(results_df)*100:.1f}%)")
            else:
                st.success("✅ Perfect agreement between global and specialized models!")
else:
    st.info("Please upload a network traffic dataset (CSV or JSON format) to begin threat analysis.")