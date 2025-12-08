"""
Data Preprocessing Module for Hybrid Classical-Quantum Fake News Detection
===========================================================================
Implements the strict pipeline: BERT Tokenization -> BERT Embeddings ([CLS]) -> PCA -> L2 Normalization
"""

import pandas as pd
import numpy as np
import torch
import os
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from typing import Tuple, Optional
import pickle
from tqdm import tqdm

# --- [FIX] Helper Function to Auto-Detect Path ---
def resolve_dataset_path(provided_path: str) -> str:
    """
    Smart logic to find the dataset regardless of what path is passed.
    Prioritizes AWS Cloud path, then Local standard path, then the provided path.
    """
    filename = os.path.basename(provided_path) # Extracts "WELFake_Dataset.csv"
    
    # 1. Check AWS Cloud Environment
    # AWS Braket sets this variable automatically
    aws_input_dir = os.environ.get("AMZN_BRAKET_INPUT_DIR")
    
    if aws_input_dir:
        # We are in the cloud. Ignore the provided path (e.g. drive/MyDrive).
        # We look in the 'dataset' channel we defined in submit_job.py
        cloud_path = os.path.join(aws_input_dir, "dataset", filename)
        
        if os.path.exists(cloud_path):
            print(f"☁️ CLOUD MODE: Overriding path. Loading from {cloud_path}")
            return cloud_path
        else:
            # Debugging helper: If file is missing, list what IS there
            print(f"⚠️ Warning: Cloud path {cloud_path} not found. Checking directory...")
            for root, dirs, files in os.walk(aws_input_dir):
                print(f"  Found in {root}: {files}")
            
    # 2. Check Local Standard Path (Good for local testing)
    local_path = os.path.join("data", filename)
    if os.path.exists(local_path):
        print(f"💻 LOCAL MODE: Found dataset at {local_path}")
        return local_path
        
    # 3. Fallback: Trust the user provided path (e.g. absolute path)
    if os.path.exists(provided_path):
        return provided_path

    # If we get here, we can't find it. Return provided_path so pandas throws the error.
    return provided_path
# ---------------------------------------------------

class BERTPCAPreprocessor:
    """
    Preprocessor following strict paper specifications:
    1. BERT tokenization (max_length=512)
    2. Extract [CLS] token embeddings (768-dim)
    3. PCA reduction to 8 dimensions
    4. L2 normalization
    """
    
    def __init__(self, n_components: int = 8, max_length: int = 512):
        self.n_components = n_components
        self.max_length = max_length
        
        # Initialize BERT tokenizer and model
        print("Loading BERT tokenizer and model (bert-base-uncased)...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        print(f"BERT model loaded on device: {self.device}")
        
        self.pca = None
        
    def extract_bert_embeddings(self, texts: list, batch_size: int = 16) -> np.ndarray:
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get BERT outputs
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Extract [CLS] token
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    def fit_transform(self, texts: list, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"\nProcessing {len(texts)} samples...")
        
        bert_embeddings = self.extract_bert_embeddings(texts)
        print(f"BERT embeddings shape: {bert_embeddings.shape}")
        
        print(f"Fitting PCA to reduce from 768 to {self.n_components} dimensions...")
        self.pca = PCA(n_components=self.n_components)
        reduced_features = self.pca.fit_transform(bert_embeddings)
        
        explained_variance = sum(self.pca.explained_variance_ratio_) * 100
        print(f"PCA explained variance: {explained_variance:.2f}%")
        
        normalized_features = normalize(reduced_features, norm='l2', axis=1)
        
        return normalized_features, labels
    
    def transform(self, texts: list) -> np.ndarray:
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_transform first.")
        
        bert_embeddings = self.extract_bert_embeddings(texts)
        reduced_features = self.pca.transform(bert_embeddings)
        normalized_features = normalize(reduced_features, norm='l2', axis=1)
        
        return normalized_features
    
    def save(self, filepath: str):
        if self.pca is None:
            raise ValueError("PCA not fitted. Nothing to save.")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'n_components': self.n_components,
                'max_length': self.max_length
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.pca = data['pca']
            self.n_components = data['n_components']
            self.max_length = data['max_length']
        print(f"Preprocessor loaded from {filepath}")


def load_welfake_dataset(filepath: str, sample_size: Optional[int] = None) -> Tuple[list, np.ndarray]:
    """Load the WELFake dataset with smart path resolution."""
    
    # [FIX] Use the smart resolver to find the real file
    actual_path = resolve_dataset_path(filepath)
    
    print(f"Loading WELFake dataset from {actual_path}...")
    df = pd.read_csv(actual_path)
    
    if 'title' in df.columns and 'text' in df.columns:
        texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()
    elif 'text' in df.columns:
        texts = df['text'].fillna('').tolist()
    else:
        raise ValueError("Dataset must contain 'text' or 'title' and 'text' columns")
    
    labels = df['label'].values
    
    if sample_size and sample_size < len(texts):
        print(f"Sampling {sample_size} examples...")
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in indices]
        labels = labels[indices]
    
    print(f"Loaded {len(texts)} samples")
    
    return texts, labels


def prepare_dataset(
    dataset_path: str,
    n_features: int = 8,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BERTPCAPreprocessor]:
    
    # Load dataset (path will be auto-corrected inside load_welfake_dataset)
    texts, labels = load_welfake_dataset(dataset_path, sample_size)
    
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    preprocessor = BERTPCAPreprocessor(n_components=n_features)
    X_train, y_train = preprocessor.fit_transform(texts_train, y_train)
    X_test = preprocessor.transform(texts_test)
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Simple test block
    print("Testing Smart Path Resolution...")
    # This simulates what run_training.py might be sending
    resolve_dataset_path("drive/MyDrive/WELFake/WELFake_Dataset.csv")