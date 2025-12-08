import os
import sys
import time
import shutil
import tempfile
from braket.aws import AwsQuantumJob
from braket.jobs.config import InstanceConfig, OutputDataConfig

# --- DEBUGGING STEP: Verify environment before submitting ---
print(f"Current Working Directory: {os.getcwd()}")
files_in_folder = os.listdir(".")
if "run_training.py" not in files_in_folder:
    print("\n❌ CRITICAL ERROR: 'run_training.py' was NOT found in this folder!")
    print(f"Files found: {files_in_folder}")
    print("Please run: 'mv src/run_training.py .' if it is still in the src folder.")
    sys.exit(1)
else:
    print("✅ Found 'run_training.py'. Proceeding...")
# ------------------------------------------------------------

# 1. Your Role ARN
my_role_arn = "arn:aws:iam::964680329916:role/AmazonBraketJobsExecutionRole-Student"

# 2. Your ECR Image URI (UPDATED to v2)
# We use v2 to ensure AWS pulls the image with the correct 'transformers' version
image_uri = "964680329916.dkr.ecr.us-east-1.amazonaws.com/hybrid-qnn-repo:v2"

# 3. Your S3 Bucket
bucket_name = "amazon-braket-hybrid-qnn-jyesht-030173" 
s3_path_string = f"s3://{bucket_name}/results"
s3_data_path = f"s3://{bucket_name}/input-data/" # Path to your uploaded CSV

print(f"Submitting job with image: {image_uri}")
print(f"Using input data from: {s3_data_path}")

# --- PREPARE SOURCE FILES ---
print("\nPreparing source files for upload...")
temp_dir = tempfile.mkdtemp(prefix="braket_job_")
print(f"Temporary directory: {temp_dir}")

# Copy only the necessary Python files
necessary_files = [
    'run_training.py',
    'quantum_model.py',
    'data_preprocessing.py',
    'requirements.txt'
]

for file in necessary_files:
    if os.path.exists(file):
        shutil.copy2(file, temp_dir)
        print(f"  ✓ Copied {file}")
    else:
        print(f"  ⚠ Warning: {file} not found")
# ----------------------------

# Define Hardware
device_config = InstanceConfig(instanceType="ml.m5.xlarge", volumeSizeInGb=30)

# Define Output Storage
output_config = OutputDataConfig(s3Path=s3_path_string)

print("\nSubmitting job to AWS Braket...")

job = AwsQuantumJob.create(
    device="local:pennylane/lightning.qubit", 
    source_module=temp_dir,      
    entry_point="run_training",           
    image_uri=image_uri,                 
    job_name=f"fake-news-qnn-{int(time.time())}",
    instance_config=device_config,
    output_data_config=output_config,
    role_arn=my_role_arn,
    
    # --- CHANGED SECTION ---
    # 1. Argument name is 'input_data'
    # 2. Format is simple: {"Channel Name": "S3 Path"}
    input_data={
        "dataset": s3_data_path
    }
    # -----------------------
)

# Cleanup temp directory
shutil.rmtree(temp_dir)
print(f"Cleaned up temporary directory")

print(f"\n✅ Job started! ARN: {job.arn}")
print(f"Check status here: https://console.aws.amazon.com/braket/home?region=us-east-1#/jobs/{job.arn}")