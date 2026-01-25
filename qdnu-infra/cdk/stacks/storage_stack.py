"""
Storage Stack - S3 Bucket for Static Content

This stack creates:
- Private S3 bucket for static web content (HTML, JS, CSS, assets)
- Bucket configured for CloudFront OAC access (no public access)
"""
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    CfnOutput,
)
from constructs import Construct


class StorageStack(Stack):
    """S3 storage for static web content."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        env_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.env_name = env_name

        # S3 bucket for static content
        # Private bucket - access only through CloudFront OAC
        self.static_bucket = s3.Bucket(
            self,
            "StaticBucket",
            bucket_name=f"{project_name}-{env_name}-static-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,  # Enable if you need versioning
            removal_policy=RemovalPolicy.RETAIN if env_name == "prod" else RemovalPolicy.DESTROY,
            auto_delete_objects=env_name != "prod",
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.HEAD],
                    allowed_origins=["*"],  # Restrict in production
                    allowed_headers=["*"],
                    max_age=3600,
                )
            ],
        )

        # Outputs
        CfnOutput(
            self,
            "StaticBucketName",
            value=self.static_bucket.bucket_name,
            description="Static content S3 bucket name",
            export_name=f"{project_name}-{env_name}-static-bucket-name",
        )

        CfnOutput(
            self,
            "StaticBucketArn",
            value=self.static_bucket.bucket_arn,
            description="Static content S3 bucket ARN",
            export_name=f"{project_name}-{env_name}-static-bucket-arn",
        )
