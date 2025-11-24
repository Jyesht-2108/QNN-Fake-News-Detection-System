"""
Data Preprocessing Module for Quantum Fake News Detection
==========================================================
This module handles loading, cleaning, and preprocessing text data for quantum neural networks.
It includes text cleaning, feature extraction (TF-IDF), and dimensionality reduction (PCA).
"""

import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle


class TextPreprocessor:
    """
    Handles text preprocessing for fake news detection.
    
    This class provides methods to clean text, extract features using TF-IDF,
    and reduce dimensionality to match quantum circuit input requirements.
    """
    
    def __init__(self, n_features: int = 8, max_tfidf_features: int = 1000):
        """
        Initialize the text preprocessor.
        
        Args:
            n_features: Number of features after PCA reduction (for quantum encoding)
            max_tfidf_features: Maximum number of TF-IDF features before PCA
        """
        self.n_features = n_features
        self.max_tfidf_features = max_tfidf_features
        self.vectorizer = None
        self.pca = None
        self.scaler = StandardScaler()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def fit_transform(self, texts: list, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform texts to quantum-ready features.
        
        Args:
            texts: List of text documents
            labels: Array of labels (0 or 1)
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        print(f"Cleaning {len(texts)} text samples...")
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Remove empty texts
        valid_indices = [i for i, text in enumerate(cleaned_texts) if text]
        cleaned_texts = [cleaned_texts[i] for i in valid_indices]
        labels = labels[valid_indices]
        
        print(f"Extracting TF-IDF features (max {self.max_tfidf_features} features)...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        tfidf_features = self.vectorizer.fit_transform(cleaned_texts).toarray()
        
        # Adjust n_components if we have fewer features than requested
        n_components = min(self.n_features, tfidf_features.shape[1], len(cleaned_texts))
        
        print(f"Reducing dimensions from {tfidf_features.shape[1]} to {n_components} using PCA...")
        self.pca = PCA(n_components=n_components)
        reduced_features = self.pca.fit_transform(tfidf_features)
        
        # Pad with zeros if we have fewer components than requested
        if reduced_features.shape[1] < self.n_features:
            padding = np.zeros((reduced_features.shape[0], self.n_features - reduced_features.shape[1]))
            reduced_features = np.hstack([reduced_features, padding])
            print(f"  Note: Padded to {self.n_features} features (had only {n_components} components)")
        
        # Normalize features to [-1, 1] range for quantum encoding
        reduced_features = self.scaler.fit_transform(reduced_features)
        
        explained_variance = sum(self.pca.explained_variance_ratio_) * 100
        print(f"PCA explained variance: {explained_variance:.2f}%")
        
        return reduced_features, labels
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform new texts using fitted preprocessor.
        
        Args:
            texts: List of text documents
            
        Returns:
            Transformed features as numpy array
        """
        if self.vectorizer is None or self.pca is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        cleaned_texts = [self.clean_text(text) for text in texts]
        tfidf_features = self.vectorizer.transform(cleaned_texts).toarray()
        reduced_features = self.pca.transform(tfidf_features)
        reduced_features = self.scaler.transform(reduced_features)
        
        return reduced_features
    
    def save(self, filepath: str):
        """Save the fitted preprocessor to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'pca': self.pca,
                'scaler': self.scaler,
                'n_features': self.n_features,
                'max_tfidf_features': self.max_tfidf_features
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a fitted preprocessor from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.pca = data['pca']
            self.scaler = data['scaler']
            self.n_features = data['n_features']
            self.max_tfidf_features = data['max_tfidf_features']
        print(f"Preprocessor loaded from {filepath}")


def load_welfake_dataset(filepath: str) -> Tuple[list, np.ndarray]:
    """
    Load the WELFake dataset.
    
    Args:
        filepath: Path to the WELFake CSV file
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading WELFake dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # WELFake format: title, text, label (0=real, 1=fake)
    # Combine title and text for richer features
    if 'title' in df.columns and 'text' in df.columns:
        texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()
    elif 'text' in df.columns:
        texts = df['text'].fillna('').tolist()
    else:
        raise ValueError("Dataset must contain 'text' or 'title' and 'text' columns")
    
    labels = df['label'].values
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: Real={sum(labels==0)}, Fake={sum(labels==1)}")
    
    return texts, labels


def load_liar_dataset(filepath: str) -> Tuple[list, np.ndarray]:
    """
    Load the LIAR dataset.
    
    Args:
        filepath: Path to the LIAR TSV file
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading LIAR dataset from {filepath}...")
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    # LIAR format: column 1 is label, column 2 is statement
    # Convert multi-class to binary: true/mostly-true/half-true -> 0 (real)
    # false/mostly-false/pants-fire -> 1 (fake)
    texts = df[2].fillna('').tolist()
    
    label_map = {
        'true': 0, 'mostly-true': 0, 'half-true': 0,
        'false': 1, 'mostly-false': 1, 'pants-fire': 1
    }
    labels = df[1].map(label_map).values
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: Real={sum(labels==0)}, Fake={sum(labels==1)}")
    
    return texts, labels


def prepare_dataset(
    dataset_path: str,
    dataset_type: str = 'welfake',
    n_features: int = 8,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TextPreprocessor]:
    """
    Complete pipeline to load and prepare dataset for quantum training.
    
    Args:
        dataset_path: Path to dataset file
        dataset_type: Type of dataset ('welfake' or 'liar')
        n_features: Number of features for quantum encoding
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        sample_size: Optional limit on number of samples (for faster testing)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Load dataset
    if dataset_type.lower() == 'welfake':
        texts, labels = load_welfake_dataset(dataset_path)
    elif dataset_type.lower() == 'liar':
        texts, labels = load_liar_dataset(dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Sample if requested (useful for quick testing)
    if sample_size and sample_size < len(texts):
        print(f"Sampling {sample_size} examples for faster processing...")
        indices = np.random.RandomState(random_state).choice(
            len(texts), sample_size, replace=False
        )
        texts = [texts[i] for i in indices]
        labels = labels[indices]
    
    # Preprocess
    preprocessor = TextPreprocessor(n_features=n_features)
    features, labels = preprocessor.fit_transform(texts, labels)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features per sample: {X_train.shape[1]}")
    print(f"  Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    """
    Example usage and testing of the preprocessing module.
    """
    # Example with synthetic data
    print("=" * 60)
    print("Testing Text Preprocessor with Synthetic Data")
    print("=" * 60)
    
    # Create sample data
    sample_texts = [
        "Breaking news: Scientists discover new quantum computing breakthrough!",
        "FAKE NEWS ALERT: Aliens landed in New York City yesterday!!!",
        "Government announces new policy on climate change regulations.",
        "You won't believe what happened next! Click here for shocking truth!",
        "Research paper published in Nature reveals important findings.",
    ]
    sample_labels = np.array([0, 1, 0, 1, 0])
    
    # Test preprocessing
    preprocessor = TextPreprocessor(n_features=4)
    features, labels = preprocessor.fit_transform(sample_texts, sample_labels)
    
    print(f"\nProcessed features shape: {features.shape}")
    print(f"Sample features:\n{features[:2]}")
    
    # Test transform on new data
    new_texts = ["This is a test article about politics."]
    new_features = preprocessor.transform(new_texts)
    print(f"\nNew sample features: {new_features}")
    
    print("\n" + "=" * 60)
    print("Preprocessing module ready!")
    print("=" * 60)
