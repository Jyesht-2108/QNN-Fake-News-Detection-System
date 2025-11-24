
> **You are my AI coding assistant. Develop a complete project for quantum neural network-based fake news detection using PennyLane (or Qiskit, if preferred). The code should be beginner-friendly, modular, and well-commented, and should run in a local Python environment. Here is my specification:**
>
> **Goal:** Build a binary classifier to detect fake news using a quantum neural network. Optionally, include adversarial robustness testing.
>
> **Detailed requirements:**
>
> 1. **Dataset:**  
> - Use the [WELFake](https://mldata.vn/english/welfake) or [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) dataset (text classification, real vs fake news).
> 
> 2. **Preprocessing:**  
> - Load the dataset and preprocess the text:  
>   - Tokenization  
>   - Lowercase  
>   - Remove punctuation/stopwords  
>   - Convert each article/headline into dense embeddings using TF-IDF or BERT (if possible)  
>   - Reduce feature size (e.g., via PCA) to match input size for quantum encoding (ideally 4–8 features).
>
> 3. **Quantum Feature Encoding:**  
> - Use amplitude encoding (preferred) or angle encoding to encode classical features into quantum states.
>
> 4. **Quantum Neural Network Model:**  
> - Build a variational quantum classifier (VQC) for binary classification:
>   - Custom quantum circuit with 2–4 variational layers
>   - Use trainable parameters
>   - Classical optimizer (Adam, etc.)
>   - Show how to combine classical preprocessing + quantum layers (hybrid model)
>
> 5. **Training and Evaluation:**  
> - Train the QNN on the dataset  
> - Output training accuracy, loss curves  
> - Evaluate on held-out test data with accuracy, precision, recall, F1-score
>
> 6. **Adversarial Robustness (if time allows):**  
> - Generate adversarial examples by modifying input text (e.g., synonym replacement with TextAttack)
>   - Retrain/test QNN on these adversarial samples
>   - Compare baseline QNN with adversarially-trained QNN performance
>
> 7. **Documentation and Comments:**  
> - Add detailed comments and docstrings to all functions
> - Provide a README with setup instructions, dependencies, and a short project summary
> - Include clear instructions on how to run each stage (preprocessing, training, evaluation)
>
> **Constraints:**  
> - Keep everything as simple and clean as possible—assume the user is new to quantum coding.
> - If any errors arise, recommend fixes.
> - At every step, log progress and print summaries.
>
> **Deliverables:**  
> - `data_preprocessing.py` (text processing and feature extraction)
> - `quantum_model.py` (quantum feature encoding and variational circuit definition)
> - `train.py` (training and evaluation loop)
> - `robustness.py` (optional: adversarial sample generation and robustness evaluation)
> - `README.md` (installation, usage, explanation)
>
> **Extra:**  
> - If possible, include plots of training curves and confusion matrix as PNG files.
>
> **Output everything in standard Python scripts ready to run, with requirements.txt.**
