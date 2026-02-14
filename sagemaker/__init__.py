"""
SageMaker infrastructure for QDNU large-scale analysis.

Components:
- train_chbmit.py: Training script for CHB-MIT analysis
- launcher.py: Job launcher with S3 integration
- requirements.txt: Container dependencies

Usage:
    # Upload data and launch job
    python sagemaker/launcher.py --action full

    # Or step by step:
    python sagemaker/launcher.py --action upload
    python sagemaker/launcher.py --action launch
    python sagemaker/launcher.py --action status
    python sagemaker/launcher.py --action download
"""
