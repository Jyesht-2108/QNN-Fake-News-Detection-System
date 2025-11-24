"""
Dataset Download Helper
======================
Helper script to download and prepare datasets for quantum fake news detection.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd


def create_data_directory():
    """Create data directory if it doesn't exist."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    print(f"Data directory created: {data_dir.absolute()}")
    return data_dir


def download_liar_dataset(data_dir: Path):
    """
    Download and extract LIAR dataset.
    
    Args:
        data_dir: Directory to save dataset
    """
    print("\n" + "=" * 60)
    print("Downloading LIAR Dataset")
    print("=" * 60)
    
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    zip_path = data_dir / "liar_dataset.zip"
    
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
        
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("Extraction complete!")
        print(f"Dataset files saved in {data_dir}")
        
        # Clean up zip file
        zip_path.unlink()
        print("Cleaned up zip file")
        
        return True
        
    except Exception as e:
        print(f"Error downloading LIAR dataset: {e}")
        return False


def create_sample_dataset(data_dir: Path, n_samples: int = 100):
    """
    Create a sample fake news dataset for testing.
    
    Args:
        data_dir: Directory to save dataset
        n_samples: Number of samples to generate
    """
    print("\n" + "=" * 60)
    print("Creating Sample Dataset")
    print("=" * 60)
    
    import random
    
    # Sample real news headlines/text
    real_samples = [
        "Scientists at MIT announce breakthrough in renewable energy research",
        "Government officials meet to discuss new climate policy initiatives",
        "Study published in Nature reveals insights into human genome",
        "Local university receives grant for cancer research program",
        "Economic report shows steady growth in manufacturing sector",
        "New archaeological discovery sheds light on ancient civilization",
        "Technology company releases quarterly earnings report",
        "International summit addresses global health challenges",
        "Research team develops new method for water purification",
        "Educational reform bill passes through legislative committee",
    ]
    
    # Sample fake news headlines/text
    fake_samples = [
        "SHOCKING: Aliens confirmed to be living among us, government admits!",
        "You won't believe this one weird trick doctors don't want you to know!",
        "BREAKING: Celebrity reveals secret to eternal youth, scientists baffled!",
        "Miracle cure discovered! Big pharma trying to hide the truth!",
        "EXPOSED: The real reason they don't want you to know this!",
        "Unbelievable footage shows mysterious creature in local park!",
        "Government conspiracy revealed! Click here for the shocking truth!",
        "This simple trick will change your life forever, guaranteed!",
        "Scientists discover earth is actually flat, mainstream media silent!",
        "Ancient prophecy predicts major event happening next week!",
    ]
    
    # Generate dataset
    data = []
    for i in range(n_samples):
        if i % 2 == 0:
            # Real news
            title = random.choice(real_samples)
            text = title + " " + " ".join(random.sample(real_samples, 3))
            label = 0
        else:
            # Fake news
            title = random.choice(fake_samples)
            text = title + " " + " ".join(random.sample(fake_samples, 3))
            label = 1
        
        data.append({
            'title': title,
            'text': text,
            'label': label
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_path = data_dir / 'Sample_Dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Created sample dataset with {n_samples} samples")
    print(f"Saved to: {output_path}")
    print(f"Label distribution: Real={sum(df['label']==0)}, Fake={sum(df['label']==1)}")
    
    return output_path


def check_existing_datasets(data_dir: Path):
    """
    Check for existing datasets in data directory.
    
    Args:
        data_dir: Directory to check
    """
    print("\n" + "=" * 60)
    print("Checking for Existing Datasets")
    print("=" * 60)
    
    datasets_found = []
    
    # Check for WELFake
    welfake_path = data_dir / 'WELFake_Dataset.csv'
    if welfake_path.exists():
        datasets_found.append(('WELFake', welfake_path))
        print(f"✓ Found WELFake dataset: {welfake_path}")
    
    # Check for LIAR
    liar_train = data_dir / 'train.tsv'
    liar_test = data_dir / 'test.tsv'
    if liar_train.exists() or liar_test.exists():
        datasets_found.append(('LIAR', data_dir))
        print(f"✓ Found LIAR dataset: {data_dir}")
    
    # Check for sample dataset
    sample_path = data_dir / 'Sample_Dataset.csv'
    if sample_path.exists():
        datasets_found.append(('Sample', sample_path))
        print(f"✓ Found Sample dataset: {sample_path}")
    
    if not datasets_found:
        print("✗ No datasets found")
    
    return datasets_found


def main():
    """
    Main function to guide user through dataset setup.
    """
    print("=" * 60)
    print("QUANTUM FAKE NEWS DETECTION - DATASET SETUP")
    print("=" * 60)
    
    # Create data directory
    data_dir = create_data_directory()
    
    # Check for existing datasets
    existing = check_existing_datasets(data_dir)
    
    if existing:
        print("\n" + "=" * 60)
        print("Datasets already available!")
        print("=" * 60)
        print("\nYou can proceed with training using:")
        for name, path in existing:
            print(f"  - {name}: {path}")
        print("\nRun: python train.py")
        return
    
    # Offer download options
    print("\n" + "=" * 60)
    print("Dataset Download Options")
    print("=" * 60)
    print("\n1. Download LIAR dataset (automatic)")
    print("2. Create sample dataset for testing (automatic)")
    print("3. Manual download instructions for WELFake")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        success = download_liar_dataset(data_dir)
        if success:
            print("\n✓ LIAR dataset ready!")
            print("Run: python train.py")
            print("(Update DATASET_TYPE='liar' in train.py)")
    
    elif choice == '2':
        sample_path = create_sample_dataset(data_dir, n_samples=200)
        print("\n✓ Sample dataset ready!")
        print("Run: python train.py")
        print(f"(Update DATASET_PATH='{sample_path}' in train.py)")
    
    elif choice == '3':
        print("\n" + "=" * 60)
        print("Manual Download Instructions - WELFake Dataset")
        print("=" * 60)
        print("\n1. Visit: https://mldata.vn/english/welfake")
        print("2. Download the WELFake dataset")
        print("3. Extract the CSV file")
        print(f"4. Place 'WELFake_Dataset.csv' in: {data_dir.absolute()}")
        print("5. Run: python train.py")
        print("\nNote: WELFake is recommended for best results")
    
    elif choice == '4':
        print("\nExiting...")
    
    else:
        print("\nInvalid choice. Please run the script again.")
    
    print("\n" + "=" * 60)
    print("Dataset setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
