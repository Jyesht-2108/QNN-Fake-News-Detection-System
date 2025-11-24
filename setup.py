"""
Setup and Installation Script
=============================
Automated setup for the quantum fake news detection project.
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def create_directories():
    """Create necessary project directories."""
    print_header("Creating Project Directories")
    
    directories = ['data', 'results', 'results/models', 'results/plots']
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    return True


def install_dependencies():
    """Install required Python packages."""
    print_header("Installing Dependencies")
    
    print("This may take several minutes...")
    print("Installing packages from requirements.txt...\n")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("\n✓ All dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        print("\nTry installing manually:")
        print("  pip install -r requirements.txt")
        return False


def download_nltk_data():
    """Download required NLTK data."""
    print_header("Downloading NLTK Data")
    
    try:
        import nltk
        
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        
        print("Downloading stopwords...")
        nltk.download('stopwords', quiet=True)
        
        print("✓ NLTK data downloaded successfully")
        return True
    
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False


def test_imports():
    """Test if all required packages can be imported."""
    print_header("Testing Package Imports")
    
    packages = [
        ('pennylane', 'PennyLane'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('torch', 'PyTorch'),
        ('nltk', 'NLTK'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    all_success = True
    
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - Failed to import")
            all_success = False
    
    if all_success:
        print("\n✓ All packages imported successfully")
    else:
        print("\n✗ Some packages failed to import")
    
    return all_success


def test_quantum_device():
    """Test if PennyLane quantum device works."""
    print_header("Testing Quantum Device")
    
    try:
        import pennylane as qml
        import numpy as np
        
        # Create a simple quantum device
        dev = qml.device('default.qubit', wires=2)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        result = circuit()
        print(f"Test circuit executed successfully")
        print(f"Result: {result}")
        print("✓ Quantum device is working")
        return True
    
    except Exception as e:
        print(f"✗ Error testing quantum device: {e}")
        return False


def run_module_tests():
    """Run basic tests on project modules."""
    print_header("Testing Project Modules")
    
    modules = [
        ('data_preprocessing.py', 'Data Preprocessing'),
        ('quantum_model.py', 'Quantum Model'),
    ]
    
    all_success = True
    
    for module_file, module_name in modules:
        try:
            print(f"\nTesting {module_name}...")
            result = subprocess.run(
                [sys.executable, module_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"✓ {module_name} test passed")
            else:
                print(f"✗ {module_name} test failed")
                print(f"Error: {result.stderr[:200]}")
                all_success = False
        
        except subprocess.TimeoutExpired:
            print(f"✗ {module_name} test timed out")
            all_success = False
        except Exception as e:
            print(f"✗ {module_name} test error: {e}")
            all_success = False
    
    return all_success


def print_next_steps():
    """Print instructions for next steps."""
    print_header("Setup Complete!")
    
    print("\n📚 Next Steps:")
    print("\n1. Download a dataset:")
    print("   python download_dataset.py")
    
    print("\n2. Train the quantum model:")
    print("   python train.py")
    
    print("\n3. Test adversarial robustness:")
    print("   python robustness.py")
    
    print("\n4. Try the interactive demo:")
    print("   python demo.py")
    
    print("\n📖 Documentation:")
    print("   Read README.md for detailed instructions")
    
    print("\n⚙️  Configuration:")
    print("   Edit config.py to customize settings")
    
    print("\n" + "=" * 60)


def main():
    """Main setup function."""
    print("=" * 60)
    print("QUANTUM FAKE NEWS DETECTION - SETUP")
    print("=" * 60)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Downloading NLTK data", download_nltk_data),
        ("Testing imports", test_imports),
        ("Testing quantum device", test_quantum_device),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
            
            if not success:
                print(f"\n⚠️  Warning: {step_name} failed")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    print("\nSetup aborted.")
                    return
        
        except Exception as e:
            print(f"\n✗ Unexpected error in {step_name}: {e}")
            results.append((step_name, False))
    
    # Print summary
    print_header("Setup Summary")
    
    for step_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n🎉 All setup steps completed successfully!")
        print_next_steps()
    else:
        print("\n⚠️  Setup completed with some warnings")
        print("You may still be able to use the project, but some features might not work.")
        print("\nCheck the errors above and try to resolve them.")


if __name__ == "__main__":
    main()
