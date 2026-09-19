"""
Braket Storage Stack.

Provisions:
- S3 results bucket (qdnu-{env}-braket-results-{account}) with partition layout
  runs/{run_id}/backend={backend}/results.parquet
- Glue database (qdnu_quantum_{env}) for Athena tables
- Athena workgroup (qdnu-braket-{env}) with a dedicated query-results bucket

The Athena tables/views themselves are NOT created here -- apply
infra/athena_ddl.sql via the Athena console once the database exists. CDK is
intentionally not the source of truth for the schema since it iterates faster
than infrastructure deploys.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_athena as athena,
    aws_glue as glue,
    aws_s3 as s3,
)
from constructs import Construct


class BraketStorageStack(Stack):
    """S3, Glue, Athena workgroup for the Braket result lake."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        env_name: str,
        module_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.env_name = env_name
        self.module_name = module_name

        # ------------------------------------------------------------------
        # Results bucket: Hybrid Jobs write Parquet under runs/{run_id}/...
        # ------------------------------------------------------------------
        is_prod = env_name == "prod"
        self.results_bucket = s3.Bucket(
            self,
            "ResultsBucket",
            bucket_name=f"{project_name}-{env_name}-{module_name}-results-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=is_prod,
            removal_policy=RemovalPolicy.RETAIN if is_prod else RemovalPolicy.DESTROY,
            auto_delete_objects=not is_prod,
            lifecycle_rules=[
                # Old experimental runs become rarely-touched after 30 days;
                # keep the data but cut storage cost.
                s3.LifecycleRule(
                    id="TransitionToIA",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        ),
                    ],
                    enabled=True,
                ),
            ],
        )

        # ------------------------------------------------------------------
        # Braket-conventional task output bucket. Required because Braket
        # validates that the s3_destination_folder for each quantum task is
        # named with the 'amazon-braket-' prefix. Used internally by
        # device.run() inside Hybrid Jobs.
        # ------------------------------------------------------------------
        self.braket_task_bucket = s3.Bucket(
            self,
            "BraketTaskBucket",
            bucket_name=f"amazon-braket-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.RETAIN if is_prod else RemovalPolicy.DESTROY,
            auto_delete_objects=not is_prod,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireTaskBlobs",
                    expiration=Duration.days(60),
                    enabled=True,
                ),
            ],
        )

        # ------------------------------------------------------------------
        # Athena query results bucket (separate; lifecycle-pruned aggressively)
        # ------------------------------------------------------------------
        self.athena_results_bucket = s3.Bucket(
            self,
            "AthenaQueryResultsBucket",
            bucket_name=f"{project_name}-{env_name}-{module_name}-athena-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireQueryResults",
                    expiration=Duration.days(14),
                    enabled=True,
                ),
            ],
        )

        # ------------------------------------------------------------------
        # Glue database for Athena tables
        # ------------------------------------------------------------------
        self.glue_database_name = f"qdnu_quantum_{env_name}"
        self.glue_database = glue.CfnDatabase(
            self,
            "GlueDatabase",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=self.glue_database_name,
                description=(
                    f"QDNU quantum result lake ({env_name}). "
                    "Tables created from infra/athena_ddl.sql."
                ),
            ),
        )

        # ------------------------------------------------------------------
        # Athena workgroup so dashboard queries are isolated from other usage
        # ------------------------------------------------------------------
        self.athena_workgroup_name = f"{project_name}-{module_name}-{env_name}"
        self.athena_workgroup = athena.CfnWorkGroup(
            self,
            "AthenaWorkgroup",
            name=self.athena_workgroup_name,
            description=f"Workgroup for {module_name} dashboards ({env_name}).",
            recursive_delete_option=not is_prod,
            state="ENABLED",
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                enforce_work_group_configuration=True,
                publish_cloud_watch_metrics_enabled=True,
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=f"s3://{self.athena_results_bucket.bucket_name}/queries/",
                    encryption_configuration=athena.CfnWorkGroup.EncryptionConfigurationProperty(
                        encryption_option="SSE_S3"
                    ),
                ),
            ),
        )

        # ------------------------------------------------------------------
        # Outputs (used by the IAM stack and by ops scripts)
        # ------------------------------------------------------------------
        CfnOutput(
            self,
            "ResultsBucketName",
            value=self.results_bucket.bucket_name,
            export_name=f"{project_name}-{env_name}-{module_name}-results-bucket",
        )
        CfnOutput(
            self,
            "ResultsBucketArn",
            value=self.results_bucket.bucket_arn,
            export_name=f"{project_name}-{env_name}-{module_name}-results-bucket-arn",
        )
        CfnOutput(
            self,
            "AthenaResultsBucketName",
            value=self.athena_results_bucket.bucket_name,
            export_name=f"{project_name}-{env_name}-{module_name}-athena-bucket",
        )
        CfnOutput(
            self,
            "BraketTaskBucketName",
            value=self.braket_task_bucket.bucket_name,
            export_name=f"{project_name}-{env_name}-{module_name}-task-bucket",
        )
        CfnOutput(
            self,
            "GlueDatabaseName",
            value=self.glue_database_name,
            export_name=f"{project_name}-{env_name}-{module_name}-glue-db",
        )
        CfnOutput(
            self,
            "AthenaWorkgroupName",
            value=self.athena_workgroup_name,
            export_name=f"{project_name}-{env_name}-{module_name}-athena-wg",
        )
