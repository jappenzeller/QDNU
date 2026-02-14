#!/usr/bin/env python3
"""
================================================================================
SAGEMAKER JOB LAUNCHER
================================================================================

Launch CHB-MIT analysis on AWS SageMaker.

Features:
- Uploads CHB-MIT data to S3 (skips existing files)
- Creates and monitors SageMaker training job
- Downloads results when complete

Usage:
    python sagemaker/launcher.py --action upload   # Upload data to S3
    python sagemaker/launcher.py --action launch   # Launch training job
    python sagemaker/launcher.py --action status   # Check job status
    python sagemaker/launcher.py --action download # Download results
    python sagemaker/launcher.py --action full     # Full pipeline

Requirements:
    pip install boto3 sagemaker

================================================================================
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import boto3
from botocore.exceptions import ClientError

# =============================================================================
# CONFIGURATION
# =============================================================================

AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
AWS_ACCOUNT_ID = os.environ.get('AWS_ACCOUNT_ID')
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'QDNU')

# S3 Configuration
S3_BUCKET = f"{PROJECT_NAME.lower()}-sagemaker-{AWS_ACCOUNT_ID}"
S3_DATA_PREFIX = "chbmit/data"
S3_OUTPUT_PREFIX = "chbmit/output"
S3_CODE_PREFIX = "chbmit/code"

# Local paths
LOCAL_DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
LOCAL_OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results" / "sagemaker"

# SageMaker Configuration
SAGEMAKER_ROLE_NAME = f"{PROJECT_NAME}-SageMakerRole"
TRAINING_INSTANCE_TYPE = "ml.m5.2xlarge"  # 8 vCPU, 32 GB RAM
TRAINING_VOLUME_SIZE = 50  # GB

# Container image (SKLearn for ML workloads)
SKLEARN_FRAMEWORK_VERSION = "1.2-1"


# =============================================================================
# AWS CLIENTS
# =============================================================================

def get_clients():
    """Get AWS clients."""
    session = boto3.Session(region_name=AWS_REGION)
    return {
        's3': session.client('s3'),
        'sagemaker': session.client('sagemaker'),
        'iam': session.client('iam'),
        'sts': session.client('sts'),
    }


# =============================================================================
# S3 OPERATIONS
# =============================================================================

def ensure_bucket(s3_client):
    """Create S3 bucket if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"Bucket exists: {S3_BUCKET}")
    except ClientError:
        print(f"Creating bucket: {S3_BUCKET}")
        if AWS_REGION == 'us-east-1':
            s3_client.create_bucket(Bucket=S3_BUCKET)
        else:
            s3_client.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
            )


def upload_data(s3_client, force: bool = False):
    """Upload CHB-MIT data to S3."""
    if not LOCAL_DATA_DIR.exists():
        print(f"Error: Data directory not found: {LOCAL_DATA_DIR}")
        return False

    ensure_bucket(s3_client)

    subjects = sorted([d for d in LOCAL_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('chb')])
    print(f"Found {len(subjects)} subjects to upload")

    uploaded = 0
    skipped = 0

    for subject_dir in subjects:
        print(f"\n  {subject_dir.name}:")

        for file_path in subject_dir.iterdir():
            if not file_path.is_file():
                continue

            s3_key = f"{S3_DATA_PREFIX}/{subject_dir.name}/{file_path.name}"

            # Check if already exists
            if not force:
                try:
                    s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                    skipped += 1
                    continue
                except ClientError:
                    pass

            # Upload
            try:
                s3_client.upload_file(str(file_path), S3_BUCKET, s3_key)
                uploaded += 1
                print(f"    Uploaded: {file_path.name}")
            except Exception as e:
                print(f"    Error uploading {file_path.name}: {e}")

    print(f"\nUpload complete: {uploaded} new, {skipped} skipped")
    return True


def upload_code(s3_client):
    """Upload training code to S3."""
    code_dir = Path(__file__).parent
    train_script = code_dir / "train_chbmit.py"
    requirements = code_dir / "requirements.txt"

    ensure_bucket(s3_client)

    for file_path in [train_script, requirements]:
        if file_path.exists():
            s3_key = f"{S3_CODE_PREFIX}/{file_path.name}"
            s3_client.upload_file(str(file_path), S3_BUCKET, s3_key)
            print(f"Uploaded: {s3_key}")


# =============================================================================
# IAM ROLE
# =============================================================================

def ensure_sagemaker_role(iam_client) -> str:
    """Create or get SageMaker execution role."""
    role_name = SAGEMAKER_ROLE_NAME

    try:
        response = iam_client.get_role(RoleName=role_name)
        print(f"Using existing role: {role_name}")
        return response['Role']['Arn']
    except ClientError:
        pass

    print(f"Creating role: {role_name}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="SageMaker execution role for QDNU project"
    )

    # Attach policies
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]

    for policy_arn in policies:
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

    # Wait for role propagation
    time.sleep(10)

    response = iam_client.get_role(RoleName=role_name)
    return response['Role']['Arn']


# =============================================================================
# SAGEMAKER JOB
# =============================================================================

def create_source_archive(s3_client) -> str:
    """Create and upload source.tar.gz for SageMaker."""
    import tarfile
    import tempfile

    code_dir = Path(__file__).parent

    # Create tar.gz
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        archive_path = tmp.name

    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(code_dir / 'train_chbmit.py', arcname='train_chbmit.py')
        if (code_dir / 'requirements.txt').exists():
            tar.add(code_dir / 'requirements.txt', arcname='requirements.txt')

    # Upload to S3
    s3_key = f"{S3_CODE_PREFIX}/sourcedir.tar.gz"
    s3_client.upload_file(archive_path, S3_BUCKET, s3_key)
    print(f"Uploaded source archive: s3://{S3_BUCKET}/{s3_key}")

    # Cleanup
    Path(archive_path).unlink()

    return f"s3://{S3_BUCKET}/{s3_key}"


def launch_training_job(sagemaker_client, iam_client, s3_client) -> str:
    """Launch SageMaker training job using boto3."""
    role_arn = ensure_sagemaker_role(iam_client)

    # Create and upload source archive
    ensure_bucket(s3_client)
    source_uri = create_source_archive(s3_client)

    job_name = f"chbmit-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # SKLearn container image
    sklearn_image = f"683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:{SKLEARN_FRAMEWORK_VERSION}-cpu-py3"

    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': role_arn,
        'AlgorithmSpecification': {
            'TrainingImage': sklearn_image,
            'TrainingInputMode': 'File',
        },
        'HyperParameters': {
            'sagemaker_program': '"train_chbmit.py"',
            'sagemaker_submit_directory': f'"{source_uri}"',
            'sagemaker_region': f'"{AWS_REGION}"',
        },
        'InputDataConfig': [
            {
                'ChannelName': 'data',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{S3_BUCKET}/{S3_DATA_PREFIX}',
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'application/octet-stream',
                'InputMode': 'File',
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': f's3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}'
        },
        'ResourceConfig': {
            'InstanceType': TRAINING_INSTANCE_TYPE,
            'InstanceCount': 1,
            'VolumeSizeInGB': TRAINING_VOLUME_SIZE,
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 86400,
        },
        'EnableManagedSpotTraining': False,
    }

    print(f"Launching training job: {job_name}")
    print(f"  Instance: {TRAINING_INSTANCE_TYPE}")
    print(f"  Image: {sklearn_image}")
    print(f"  Data: s3://{S3_BUCKET}/{S3_DATA_PREFIX}")

    sagemaker_client.create_training_job(**training_params)

    # Save job info
    job_info = {
        'job_name': job_name,
        'created': datetime.now().isoformat(),
        's3_bucket': S3_BUCKET,
        's3_output': f's3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/{job_name}',
    }

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_OUTPUT_DIR / 'last_job.json', 'w') as f:
        json.dump(job_info, f, indent=2)

    print(f"\nJob launched successfully!")
    print(f"  Job name: {job_name}")
    print(f"  Output: {job_info['s3_output']}")

    return job_name


def get_job_status(sagemaker_client, job_name: str = None) -> dict:
    """Get status of training job."""
    if not job_name:
        job_file = LOCAL_OUTPUT_DIR / 'last_job.json'
        if job_file.exists():
            with open(job_file) as f:
                job_info = json.load(f)
                job_name = job_info['job_name']
        else:
            print("No job name provided and no last_job.json found")
            return None

    response = sagemaker_client.describe_training_job(TrainingJobName=job_name)

    status = {
        'job_name': job_name,
        'status': response['TrainingJobStatus'],
        'secondary_status': response.get('SecondaryStatus', ''),
        'creation_time': response['CreationTime'].isoformat(),
    }

    if 'TrainingEndTime' in response:
        status['end_time'] = response['TrainingEndTime'].isoformat()

    if response['TrainingJobStatus'] == 'Failed':
        status['failure_reason'] = response.get('FailureReason', 'Unknown')

    return status


def wait_for_job(sagemaker_client, job_name: str):
    """Wait for training job to complete."""
    print(f"Waiting for job: {job_name}")

    while True:
        status = get_job_status(sagemaker_client, job_name)
        print(f"  Status: {status['status']} - {status['secondary_status']}")

        if status['status'] in ['Completed', 'Failed', 'Stopped']:
            return status

        time.sleep(60)


def download_results(s3_client, job_name: str = None):
    """Download results from S3."""
    if not job_name:
        job_file = LOCAL_OUTPUT_DIR / 'last_job.json'
        if job_file.exists():
            with open(job_file) as f:
                job_info = json.load(f)
                job_name = job_info['job_name']
        else:
            print("No job name provided")
            return

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s3_prefix = f"{S3_OUTPUT_PREFIX}/{job_name}"

    print(f"Downloading results from s3://{S3_BUCKET}/{s3_prefix}")

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            local_path = LOCAL_OUTPUT_DIR / Path(key).name

            print(f"  Downloading: {Path(key).name}")
            s3_client.download_file(S3_BUCKET, key, str(local_path))

    print(f"\nResults saved to: {LOCAL_OUTPUT_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SageMaker Job Launcher")
    parser.add_argument('--action', choices=['upload', 'launch', 'status', 'download', 'full', 'wait'],
                        default='status', help='Action to perform')
    parser.add_argument('--job-name', type=str, help='Training job name')
    parser.add_argument('--force', action='store_true', help='Force re-upload of data')

    args = parser.parse_args()

    clients = get_clients()

    if args.action == 'upload':
        upload_data(clients['s3'], force=args.force)

    elif args.action == 'launch':
        launch_training_job(clients['sagemaker'], clients['iam'], clients['s3'])

    elif args.action == 'status':
        status = get_job_status(clients['sagemaker'], args.job_name)
        if status:
            print(json.dumps(status, indent=2))

    elif args.action == 'wait':
        job_file = LOCAL_OUTPUT_DIR / 'last_job.json'
        if job_file.exists():
            with open(job_file) as f:
                job_name = json.load(f)['job_name']
            status = wait_for_job(clients['sagemaker'], job_name)
            print(json.dumps(status, indent=2))

    elif args.action == 'download':
        download_results(clients['s3'], args.job_name)

    elif args.action == 'full':
        print("=" * 80)
        print("FULL SAGEMAKER PIPELINE")
        print("=" * 80)

        print("\n[1/4] Uploading data to S3...")
        upload_data(clients['s3'])

        print("\n[2/4] Launching training job...")
        job_name = launch_training_job(clients['sagemaker'], clients['iam'], clients['s3'])

        print("\n[3/4] Waiting for job completion...")
        status = wait_for_job(clients['sagemaker'], job_name)

        if status['status'] == 'Completed':
            print("\n[4/4] Downloading results...")
            download_results(clients['s3'], job_name)
            print("\nPipeline completed successfully!")
        else:
            print(f"\nJob failed: {status.get('failure_reason', 'Unknown')}")


if __name__ == '__main__':
    main()
