"""
Configuration File
=================
Central configuration for the quantum fake news detection project.
"""

from pathlib import Path


class Config:
    """
    Configuration class for quantum fake news detector.
    
    Modify these settings to customize your training and model.
    """
    
    # ==================== DATA CONFIGURATION ====================
    
    # Dataset settings
    DATASET_PATH = 'data/WELFake_Dataset.csv'  # Path to dataset
    DATASET_TYPE = 'welfake'  # 'welfake' or 'liar'
    
    # Sampling (set to None to use full dataset)
    SAMPLE_SIZE = 1000  # Number of samples to use (for faster testing)
    
    # Train/test split
    TEST_SIZE = 0.2  # Fraction of data for testing
    RANDOM_STATE = 42  # Random seed for reproducibility
    
    # ==================== PREPROCESSING CONFIGURATION ====================
    
    # Feature extraction
    N_FEATURES = 8  # Number of features after PCA (must match N_QUBITS)
    MAX_TFIDF_FEATURES = 1000  # Maximum TF-IDF features before PCA
    
    # ==================== QUANTUM MODEL CONFIGURATION ====================
    
    # Quantum circuit architecture
    N_QUBITS = 8  # Number of qubits (should match N_FEATURES)
    N_LAYERS = 3  # Number of variational layers
    DEVICE_NAME = 'default.qubit'  # PennyLane device
    
    # Encoding method
    ENCODING_METHOD = 'amplitude'  # 'amplitude' or 'angle'
    
    # ==================== TRAINING CONFIGURATION ====================
    
    # Training hyperparameters
    EPOCHS = 50  # Number of training epochs
    BATCH_SIZE = 10  # Batch size for training
    LEARNING_RATE = 0.01  # Learning rate
    OPTIMIZER = 'adam'  # 'adam', 'sgd', or 'adagrad'
    
    # Early stopping
    EARLY_STOPPING = False  # Enable early stopping
    PATIENCE = 10  # Epochs to wait before stopping
    
    # ==================== EVALUATION CONFIGURATION ====================
    
    # Classification threshold
    CLASSIFICATION_THRESHOLD = 0.5  # Threshold for binary classification
    
    # ==================== ROBUSTNESS CONFIGURATION ====================
    
    # Adversarial testing
    ATTACK_TYPES = ['synonym', 'char_swap', 'deletion', 'insertion', 'mixed']
    
    # Adversarial training
    ADVERSARIAL_TRAINING = False  # Enable adversarial training
    AUGMENTATION_RATIO = 0.3  # Fraction of adversarial examples to add
    
    # ==================== OUTPUT CONFIGURATION ====================
    
    # Directories
    DATA_DIR = Path('data')
    RESULTS_DIR = Path('results')
    MODELS_DIR = RESULTS_DIR / 'models'
    PLOTS_DIR = RESULTS_DIR / 'plots'
    
    # File names
    MODEL_FILE = 'quantum_model.pkl'
    PREPROCESSOR_FILE = 'preprocessor.pkl'
    METRICS_FILE = 'metrics.json'
    
    # Visualization
    SAVE_PLOTS = True  # Save plots to disk
    SHOW_PLOTS = True  # Display plots
    PLOT_DPI = 300  # Plot resolution
    
    # ==================== LOGGING CONFIGURATION ====================
    
    # Verbosity
    VERBOSE = True  # Print detailed progress
    LOG_INTERVAL = 10  # Print every N epochs
    
    # ==================== ADVANCED CONFIGURATION ====================
    
    # Quantum simulation
    SHOTS = None  # Number of shots (None = exact simulation)
    
    # Gradient computation
    GRADIENT_METHOD = 'parameter-shift'  # 'parameter-shift' or 'finite-diff'
    
    # Numerical stability
    EPSILON = 1e-7  # Small constant for numerical stability
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"\nData:")
        print(f"  Dataset: {cls.DATASET_PATH}")
        print(f"  Type: {cls.DATASET_TYPE}")
        print(f"  Sample size: {cls.SAMPLE_SIZE or 'Full dataset'}")
        
        print(f"\nPreprocessing:")
        print(f"  Features: {cls.N_FEATURES}")
        print(f"  TF-IDF features: {cls.MAX_TFIDF_FEATURES}")
        
        print(f"\nQuantum Model:")
        print(f"  Qubits: {cls.N_QUBITS}")
        print(f"  Layers: {cls.N_LAYERS}")
        print(f"  Encoding: {cls.ENCODING_METHOD}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Optimizer: {cls.OPTIMIZER}")
        
        print(f"\nOutput:")
        print(f"  Results directory: {cls.RESULTS_DIR}")
        print("=" * 60)
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        errors = []
        
        # Check if qubits match features
        if cls.N_QUBITS != cls.N_FEATURES:
            errors.append(f"N_QUBITS ({cls.N_QUBITS}) should match N_FEATURES ({cls.N_FEATURES})")
        
        # Check if dataset exists
        if not Path(cls.DATASET_PATH).exists() and cls.SAMPLE_SIZE is None:
            print(f"Warning: Dataset not found at {cls.DATASET_PATH}")
            print("Will use synthetic data for demonstration.")
        
        # Check valid optimizer
        if cls.OPTIMIZER not in ['adam', 'sgd', 'adagrad']:
            errors.append(f"Invalid optimizer: {cls.OPTIMIZER}")
        
        # Check valid encoding
        if cls.ENCODING_METHOD not in ['amplitude', 'angle']:
            errors.append(f"Invalid encoding method: {cls.ENCODING_METHOD}")
        
        if errors:
            print("\n⚠️  Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Quick access to common configurations

class QuickConfig:
    """Pre-defined configurations for common use cases."""
    
    @staticmethod
    def fast_testing():
        """Configuration for fast testing (small model, few samples)."""
        Config.SAMPLE_SIZE = 200
        Config.N_FEATURES = 4
        Config.N_QUBITS = 4
        Config.N_LAYERS = 2
        Config.EPOCHS = 20
        Config.BATCH_SIZE = 10
        print("✓ Fast testing configuration loaded")
    
    @staticmethod
    def full_training():
        """Configuration for full training (larger model, all data)."""
        Config.SAMPLE_SIZE = None  # Use all data
        Config.N_FEATURES = 8
        Config.N_QUBITS = 8
        Config.N_LAYERS = 3
        Config.EPOCHS = 100
        Config.BATCH_SIZE = 10
        print("✓ Full training configuration loaded")
    
    @staticmethod
    def high_accuracy():
        """Configuration optimized for accuracy."""
        Config.SAMPLE_SIZE = None
        Config.N_FEATURES = 16
        Config.N_QUBITS = 16
        Config.N_LAYERS = 4
        Config.EPOCHS = 150
        Config.BATCH_SIZE = 5
        Config.LEARNING_RATE = 0.005
        print("✓ High accuracy configuration loaded")
    
    @staticmethod
    def adversarial_robust():
        """Configuration for adversarial robustness."""
        Config.ADVERSARIAL_TRAINING = True
        Config.AUGMENTATION_RATIO = 0.3
        Config.EPOCHS = 80
        print("✓ Adversarial robustness configuration loaded")


if __name__ == "__main__":
    """Test configuration."""
    print("Testing configuration...")
    
    Config.print_config()
    
    print("\nValidating configuration...")
    if Config.validate():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
    
    print("\nCreating directories...")
    Config.create_directories()
    print("✓ Directories created")
    
    print("\n" + "=" * 60)
    print("Quick Configurations Available:")
    print("=" * 60)
    print("1. QuickConfig.fast_testing() - For quick testing")
    print("2. QuickConfig.full_training() - For full training")
    print("3. QuickConfig.high_accuracy() - For best accuracy")
    print("4. QuickConfig.adversarial_robust() - For robustness")
