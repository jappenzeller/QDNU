"""
Braket IAM Stack.

Provisions the execution role assumed by Braket Hybrid Jobs. The role:
- trusts braket.amazonaws.com
- can run quantum tasks against any Braket device
- can read input data and write Parquet results to the results bucket
- can write CloudWatch logs and metrics
- can pull the default Hybrid Jobs container image from Braket's ECR repos
- attaches the AWS-managed AmazonBraketJobsExecutionPolicy as a baseline
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


class BraketIamStack(Stack):
    """Hybrid Jobs execution role for the Braket project."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        env_name: str,
        module_name: str,
        results_bucket: s3.IBucket,
        task_bucket: s3.IBucket | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        role_name = f"{project_name}-{env_name}-{module_name}-jobs-role"
        self.jobs_role = iam.Role(
            self,
            "HybridJobsRole",
            role_name=role_name,
            assumed_by=iam.ServicePrincipal("braket.amazonaws.com"),
            description=(
                "Execution role for Braket Hybrid Jobs in the QDNU "
                f"{module_name} pipeline ({env_name})."
            ),
            managed_policies=[
                # AWS-managed baseline for Hybrid Jobs (Braket task perms,
                # CloudWatch Logs, ECR pull, S3 access to Braket-managed paths).
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonBraketJobsExecutionPolicy"
                ),
            ],
        )

        # Read job input + write Parquet results to our specific bucket.
        # The managed policy already covers Braket-managed buckets, but our
        # results bucket is named explicitly so we grant on top.
        results_bucket.grant_read_write(self.jobs_role)

        # The amazon-braket-prefixed task bucket is also explicit; grant in
        # case the managed policy's bucket-name pattern does not match.
        if task_bucket is not None:
            task_bucket.grant_read_write(self.jobs_role)

        # Allow Braket task creation/inspection across all device ARNs.
        self.jobs_role.add_to_policy(
            iam.PolicyStatement(
                sid="BraketTaskAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "braket:CreateQuantumTask",
                    "braket:GetQuantumTask",
                    "braket:CancelQuantumTask",
                    "braket:SearchQuantumTasks",
                    "braket:GetDevice",
                    "braket:SearchDevices",
                ],
                resources=["*"],
            )
        )

        # CloudWatch metrics (the managed policy covers logs but not metrics).
        self.jobs_role.add_to_policy(
            iam.PolicyStatement(
                sid="EmitCustomMetrics",
                effect=iam.Effect.ALLOW,
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "cloudwatch:namespace": [
                            "/aws/braket",
                            f"qdnu/{module_name}",
                        ]
                    }
                },
            )
        )

        CfnOutput(
            self,
            "HybridJobsRoleArn",
            value=self.jobs_role.role_arn,
            description="ARN to pass to AwsQuantumJob.create(role_arn=...)",
            export_name=f"{project_name}-{env_name}-{module_name}-jobs-role-arn",
        )
        CfnOutput(
            self,
            "HybridJobsRoleName",
            value=self.jobs_role.role_name,
            export_name=f"{project_name}-{env_name}-{module_name}-jobs-role-name",
        )
