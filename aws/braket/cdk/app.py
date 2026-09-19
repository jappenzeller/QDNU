#!/usr/bin/env python3
"""
QDNU Braket Infrastructure - AWS CDK Application.

Stacks:
- BraketStorageStack: S3 result bucket, Glue database, Athena workgroup
- BraketIamStack: Hybrid Jobs execution role (consumes the storage stack)

Deploy with:
    cdk bootstrap
    cdk deploy --all

The stack is intentionally region-pinned. Braket QPU access depends on the
region you submit Hybrid Jobs from (IonQ in us-east-1, Rigetti in us-west-1,
IQM in eu-north-1, simulators everywhere). Default region is us-east-1; set
CDK_DEFAULT_REGION to deploy elsewhere or stand up parallel result buckets.
"""
from __future__ import annotations

import os

import aws_cdk as cdk

from stacks import BraketIamStack, BraketStorageStack


def main() -> None:
    app = cdk.App()

    project_name = app.node.try_get_context("project_name") or "qdnu"
    env_name = app.node.try_get_context("environment") or "dev"
    module_name = app.node.try_get_context("module_name") or "braket"

    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    )

    common_tags = {
        "Project": project_name,
        "Module": module_name,
        "Environment": env_name,
        "ManagedBy": "CDK",
    }

    def stack_name(suffix: str) -> str:
        return f"{project_name}-{env_name}-{module_name}-{suffix}"

    storage = BraketStorageStack(
        app,
        stack_name("storage"),
        env=env,
        project_name=project_name,
        env_name=env_name,
        module_name=module_name,
    )

    iam = BraketIamStack(
        app,
        stack_name("iam"),
        env=env,
        project_name=project_name,
        env_name=env_name,
        module_name=module_name,
        results_bucket=storage.results_bucket,
        task_bucket=storage.braket_task_bucket,
    )
    iam.add_dependency(storage)

    for stack in (storage, iam):
        for k, v in common_tags.items():
            cdk.Tags.of(stack).add(k, v)

    app.synth()


if __name__ == "__main__":
    main()
