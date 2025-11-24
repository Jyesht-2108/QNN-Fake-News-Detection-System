"""
Adversarial Robustness Testing for Quantum Fake News Detector
=============================================================
This module tests the quantum model's robustness against adversarial text attacks.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score
import random

from quantum_model import QuantumNeuralNetwork
from data_preprocessing import TextPreprocessor


class SimpleTextAttacker:
    """
    Simple adversarial text attack generator.
    
    This class implements basic text perturbation techniques:
    - Synonym replacement
    - Character swapping
    - Word insertion/deletion
    """
    
    def __init__(self):
        """Initialize the text attacker with common synonyms."""
        # Simple synonym dictionary for demonstration
        self.synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'poor', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'giant'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'new': ['recent', 'latest', 'fresh', 'modern'],
            'old': ['ancient', 'aged', 'vintage', 'antique'],
            'important': ['crucial', 'vital', 'significant', 'essential'],
            'said': ['stated', 'mentioned', 'declared', 'announced'],
            'people': ['individuals', 'persons', 'citizens', 'folks'],
            'government': ['administration', 'authorities', 'officials', 'state'],
        }
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """
        Replace n words with their synonyms.
        
        Args:
            text: Original text
            n: Number of words to replace
            
        Returns:
            Perturbed text
        """
        words = text.split()
        replaceable = [i for i, word in enumerate(words) if word.lower() in self.synonyms]
        
        if not replaceable:
            return text
        
        # Randomly select words to replace
        n_replace = min(n, len(replaceable))
        indices = random.sample(replaceable, n_replace)
        
        for idx in indices:
            word = words[idx].lower()
            if word in self.synonyms:
                words[idx] = random.choice(self.synonyms[word])
        
        return ' '.join(words)
    
    def character_swap(self, text: str, n: int = 2) -> str:
        """
        Swap adjacent characters in n random positions.
        
        Args:
            text: Original text
            n: Number of swaps
            
        Returns:
            Perturbed text
        """
        chars = list(text)
        swappable = list(range(len(chars) - 1))
        
        if not swappable:
            return text
        
        n_swap = min(n, len(swappable))
        indices = random.sample(swappable, n_swap)
        
        for idx in indices:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        
        return ''.join(chars)
    
    def word_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p.
        
        Args:
            text: Original text
            p: Deletion probability
            
        Returns:
            Perturbed text
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Keep at least one word
        new_words = [word for word in words if random.random() > p]
        if not new_words:
            new_words = [random.choice(words)]
        
        return ' '.join(new_words)
    
    def word_insertion(self, text: str, n: int = 1) -> str:
        """
        Insert random words from the text at random positions.
        
        Args:
            text: Original text
            n: Number of insertions
            
        Returns:
            Perturbed text
        """
        words = text.split()
        if not words:
            return text
        
        for _ in range(n):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def generate_adversarial(self, text: str, attack_type: str = 'mixed') -> str:
        """
        Generate adversarial example using specified attack.
        
        Args:
            text: Original text
            attack_type: Type of attack ('synonym', 'char_swap', 'deletion', 'insertion', 'mixed')
            
        Returns:
            Adversarial text
        """
        if attack_type == 'synonym':
            return self.synonym_replacement(text, n=2)
        elif attack_type == 'char_swap':
            return self.character_swap(text, n=2)
        elif attack_type == 'deletion':
            return self.word_deletion(text, p=0.1)
        elif attack_type == 'insertion':
            return self.word_insertion(text, n=1)
        elif attack_type == 'mixed':
            # Apply multiple attacks
            text = self.synonym_replacement(text, n=1)
            text = self.character_swap(text, n=1)
            return text
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")


class RobustnessTester:
    """
    Test quantum model robustness against adversarial attacks.
    """
    
    def __init__(
        self,
        qnn: QuantumNeuralNetwork,
        preprocessor: TextPreprocessor,
        attacker: SimpleTextAttacker = None
    ):
        """
        Initialize robustness tester.
        
        Args:
            qnn: Trained quantum neural network
            preprocessor: Fitted text preprocessor
            attacker: Text attack generator
        """
        self.qnn = qnn
        self.preprocessor = preprocessor
        self.attacker = attacker or SimpleTextAttacker()
    
    def test_robustness(
        self,
        texts: List[str],
        labels: np.ndarray,
        attack_types: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Test model robustness against various attacks.
        
        Args:
            texts: Original text samples
            labels: True labels
            attack_types: List of attack types to test
            
        Returns:
            Dictionary of results for each attack type
        """
        if attack_types is None:
            attack_types = ['synonym', 'char_swap', 'deletion', 'insertion', 'mixed']
        
        print("\n" + "=" * 60)
        print("ADVERSARIAL ROBUSTNESS TESTING")
        print("=" * 60)
        
        # Get baseline accuracy on clean data
        print("\nProcessing clean samples...")
        clean_features = self.preprocessor.transform(texts)
        clean_preds = self.qnn.predict_batch(clean_features)
        baseline_acc = accuracy_score(labels, clean_preds)
        
        print(f"Baseline accuracy (clean data): {baseline_acc:.4f}")
        
        results = {'clean': {'accuracy': baseline_acc, 'samples': len(texts)}}
        
        # Test each attack type
        for attack_type in attack_types:
            print(f"\nTesting {attack_type} attack...")
            
            # Generate adversarial examples
            adv_texts = [self.attacker.generate_adversarial(text, attack_type) 
                        for text in texts]
            
            # Process and predict
            try:
                adv_features = self.preprocessor.transform(adv_texts)
                adv_preds = self.qnn.predict_batch(adv_features)
                adv_acc = accuracy_score(labels, adv_preds)
                
                # Calculate accuracy drop
                acc_drop = baseline_acc - adv_acc
                
                results[attack_type] = {
                    'accuracy': adv_acc,
                    'accuracy_drop': acc_drop,
                    'samples': len(adv_texts)
                }
                
                print(f"  Accuracy: {adv_acc:.4f}")
                print(f"  Accuracy drop: {acc_drop:.4f} ({acc_drop/baseline_acc*100:.1f}%)")
                
            except Exception as e:
                print(f"  Error during {attack_type} attack: {e}")
                results[attack_type] = {'error': str(e)}
        
        return results
    
    def plot_robustness_results(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: str = None
    ):
        """
        Visualize robustness test results.
        
        Args:
            results: Results from test_robustness
            save_path: Optional path to save plot
        """
        # Extract data for plotting
        attack_names = []
        accuracies = []
        
        for attack_type, metrics in results.items():
            if 'accuracy' in metrics:
                attack_names.append(attack_type.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if name == 'Clean' else 'orange' for name in attack_names]
        bars = ax.bar(attack_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_title('Quantum Model Robustness Against Adversarial Attacks', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nRobustness plot saved to {save_path}")
        
        plt.show()


def adversarial_training(
    qnn: QuantumNeuralNetwork,
    preprocessor: TextPreprocessor,
    texts_train: List[str],
    labels_train: np.ndarray,
    texts_val: List[str],
    labels_val: np.ndarray,
    epochs: int = 20,
    augmentation_ratio: float = 0.3
) -> QuantumNeuralNetwork:
    """
    Train model with adversarial examples for improved robustness.
    
    Args:
        qnn: Quantum neural network
        preprocessor: Text preprocessor
        texts_train: Training texts
        labels_train: Training labels
        texts_val: Validation texts
        labels_val: Validation labels
        epochs: Number of training epochs
        augmentation_ratio: Fraction of adversarial examples to add
        
    Returns:
        Trained quantum neural network
    """
    print("\n" + "=" * 60)
    print("ADVERSARIAL TRAINING")
    print("=" * 60)
    
    attacker = SimpleTextAttacker()
    
    # Generate adversarial examples
    n_adv = int(len(texts_train) * augmentation_ratio)
    print(f"\nGenerating {n_adv} adversarial examples...")
    
    adv_indices = np.random.choice(len(texts_train), n_adv, replace=False)
    adv_texts = [attacker.generate_adversarial(texts_train[i], 'mixed') 
                for i in adv_indices]
    adv_labels = labels_train[adv_indices]
    
    # Combine clean and adversarial data
    combined_texts = list(texts_train) + adv_texts
    combined_labels = np.concatenate([labels_train, adv_labels])
    
    print(f"Total training samples: {len(combined_texts)} (clean + adversarial)")
    
    # Preprocess combined data
    combined_features = preprocessor.transform(combined_texts)
    val_features = preprocessor.transform(texts_val)
    
    # Train with adversarial examples
    from train import QuantumTrainer
    
    trainer = QuantumTrainer(qnn, learning_rate=0.01, optimizer='adam')
    trainer.train(
        X_train=combined_features,
        y_train=combined_labels,
        X_val=val_features,
        y_val=labels_val,
        epochs=epochs,
        batch_size=10,
        verbose=True
    )
    
    print("\nAdversarial training complete!")
    
    return qnn


def main():
    """
    Main robustness testing pipeline.
    """
    print("=" * 60)
    print("QUANTUM FAKE NEWS DETECTOR - ROBUSTNESS TESTING")
    print("=" * 60)
    
    # Load trained model and preprocessor
    print("\nLoading trained model and preprocessor...")
    
    try:
        qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=3)
        qnn.load('results/quantum_model.pkl')
        
        preprocessor = TextPreprocessor(n_features=8)
        preprocessor.load('results/preprocessor.pkl')
        
        print("Model and preprocessor loaded successfully!")
        
    except FileNotFoundError:
        print("Error: Trained model not found. Please run train.py first.")
        return
    
    # Create test samples
    print("\nCreating test samples...")
    test_texts = [
        "Scientists announce breakthrough in quantum computing research at major university.",
        "SHOCKING: Aliens spotted in downtown area, government hiding the truth!",
        "New policy announced by government officials regarding climate change initiatives.",
        "You won't believe what this celebrity did! Click here for exclusive photos!",
        "Research study published in peer-reviewed journal shows promising results.",
        "BREAKING: Miracle cure discovered, doctors don't want you to know!",
    ]
    test_labels = np.array([0, 1, 0, 1, 0, 1])  # 0=real, 1=fake
    
    # Test robustness
    tester = RobustnessTester(qnn, preprocessor)
    results = tester.test_robustness(test_texts, test_labels)
    
    # Visualize results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    tester.plot_robustness_results(results, save_path='results/robustness_results.png')
    
    # Save results
    import json
    with open('results/robustness_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nRobustness metrics saved to results/robustness_metrics.json")
    
    print("\n" + "=" * 60)
    print("ROBUSTNESS TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
