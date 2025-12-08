# Use an official PyTorch image as the base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system tools
RUN apt-get update && apt-get install -y git

# Install the Python libraries your project needs
RUN pip install --no-cache-dir \
    pennylane \
    transformers==4.35.2 \
    scikit-learn \
    pandas \
    amazon-braket-sdk \
    amazon-braket-pennylane-plugin \
    boto3 \
    matplotlib \
    seaborn

# Set the working directory inside the container
WORKDIR /opt/ml/code

# Copy all your project files into the container
COPY . /opt/ml/code

# specific instructions for AWS to run your training script
ENTRYPOINT ["python", "run_training.py"]
