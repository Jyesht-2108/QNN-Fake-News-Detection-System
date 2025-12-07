"""
Data Preprocessing Module for Hybrid Classical-Quantum Fake News Detection
===========================================================================
Implements the strict pipeline: BERT Tokenization -> BERT Embeddings ([CLS]) -> PCA -> L2 Normalization
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from typing import Tuple, Optional
import pickle
from tqdm import tqdm


class BERTPCAPreprocessor:
    """
    Preprocessor following strict paper specifications:
    1. BERT tokenization (max_length=512)
    2. Extract [CLS] token embeddings (768-dim)
    3. PCA reduction to 8 dimensions
    4. L2 normalization
    """
    
    def __init__(self, n_components: int = 8, max_length: int = 512):
        """
        Initialize preprocessor.
        
        Args:
            n_components: Target dimensions after PCA (default: 8)
            max_length: Max sequence length for BERT (default: 512)
        """
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
        
        # PCA will be fitted later
        self.pca = None
        
    def extract_bert_embeddings(self, texts: list, batch_size: int = 16) -> np.ndarray:
        """
        Extract [CLS] token embeddings from BERT for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of shape (n_samples, 768) containing [CLS] embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize with padding and truncation
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
                
                # Extract [CLS] token (first token) from last hidden state
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    def fit_transform(self, texts: list, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA and transform texts to quantum-ready features.
        
        Args:
            texts: List of text documents
            labels: Array of labels (0 or 1)
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        print(f"\nProcessing {len(texts)} samples...")
        
        # Step 1: Extract BERT embeddings (768-dim)
        bert_embeddings = self.extract_bert_embeddings(texts)
        print(f"BERT embeddings shape: {bert_embeddings.shape}")
        
        # Step 2: Fit PCA and reduce to n_components dimensions
        print(f"Fitting PCA to reduce from 768 to {self.n_components} dimensions...")
        self.pca = PCA(n_components=self.n_components)
        reduced_features = self.pca.fit_transform(bert_embeddings)
        
        explained_variance = sum(self.pca.explained_variance_ratio_) * 100
        print(f"PCA explained variance: {explained_variance:.2f}%")
        
        # Step 3: L2 normalization (unit norm)
        normalized_features = normalize(reduced_features, norm='l2', axis=1)
        print(f"Final features shape: {normalized_features.shape}")
        print(f"Feature range: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")
        
        return normalized_features, labels
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform new texts using fitted PCA.
        
        Args:
            texts: List of text documents
            
        Returns:
            Transformed and normalized features
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_transform first.")
        
        # Extract BERT embeddings
        bert_embeddings = self.extract_bert_embeddings(texts)
        
        # Apply PCA transformation
        reduced_features = self.pca.transform(bert_embeddings)
        
        # L2 normalization
        normalized_features = normalize(reduced_features, norm='l2', axis=1)
        
        return normalized_features
    
    def save(self, filepath: str):
        """Save the fitted preprocessor (PCA only, BERT is reloaded)."""
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
        """Load a fitted preprocessor."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.pca = data['pca']
            self.n_components = data['n_components']
            self.max_length = data['max_length']
        print(f"Preprocessor loaded from {filepath}")


def load_welfake_dataset(filepath: str, sample_size: Optional[int] = None) -> Tuple[list, np.ndarray]:
    """
    Load the WELFake dataset.
    
    Args:
        filepath: Path to the WELFake CSV file
        sample_size: Optional limit on number of samples
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading WELFake dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Combine title and text
    if 'title' in df.columns and 'text' in df.columns:
        texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()
    elif 'text' in df.columns:
        texts = df['text'].fillna('').tolist()
    else:
        raise ValueError("Dataset must contain 'text' or 'title' and 'text' columns")
    
    labels = df['label'].values
    
    # Sample if requested
    if sample_size and sample_size < len(texts):
        print(f"Sampling {sample_size} examples...")
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in indices]
        labels = labels[indices]
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: Real={sum(labels==0)}, Fake={sum(labels==1)}")
    
    return texts, labels


def prepare_dataset(
    dataset_path: str,
    n_features: int = 8,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BERTPCAPreprocessor]:
    """
    Complete preprocessing pipeline.
    
    Args:
        dataset_path: Path to WELFake dataset
        n_features: Number of features after PCA (default: 8)
        test_size: Test split ratio
        random_state: Random seed
        sample_size: Optional sample limit
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Load dataset
    texts, labels = load_welfake_dataset(dataset_path, sample_size)
    
    # Split before preprocessing to avoid data leakage
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Initialize preprocessor
    preprocessor = BERTPCAPreprocessor(n_components=n_features)
    
    # Fit on training data and transform
    X_train, y_train = preprocessor.fit_transform(texts_train, y_train)
    
    # Transform test data
    X_test = preprocessor.transform(texts_test)
    
    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features per sample: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test with sample data
    print("Testing BERT-PCA Preprocessor...")
    
    sample_texts = [
        "Breaking news: Scientists discover new quantum computing breakthrough!",
        "FAKE NEWS ALERT: Aliens landed in New York City yesterday!!!",
        "Government announces new policy on climate change regulations."
    ]
    sample_labels = np.array([0, 1, 0])
    
    preprocessor = BERTPCAPreprocessor(n_components=8)
    features, labels = preprocessor.fit_transform(sample_texts, sample_labels)
    
    print(f"\nProcessed features shape: {features.shape}")
    print(f"Sample features:\n{features}")
