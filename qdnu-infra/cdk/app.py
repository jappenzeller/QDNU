#!/usr/bin/env python3
"""
QDNU Infrastructure - AWS CDK Application

Hybrid static/dynamic web application infrastructure:
- CloudFront as single entry point
- S3 for static content
- API Gateway + Lambda for dynamic API
- ALB/NLB for routing
"""
import os
import aws_cdk as cdk
from stacks import (
    NetworkStack,
    StorageStack,
    ComputeStack,
    ApiStack,
    LoadBalancerStack,
    CdnStack,
)


def main():
    app = cdk.App()

    # Get configuration from context
    env_name = app.node.try_get_context("environment") or "dev"
    project_name = app.node.try_get_context("project_name") or "qdnu"
    domain_name = app.node.try_get_context("domain_name") or ""

    # Environment configuration
    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    )

    # Common tags for all resources
    common_tags = {
        "Project": project_name,
        "Environment": env_name,
        "ManagedBy": "CDK",
    }

    # Stack naming convention
    def stack_name(name: str) -> str:
        return f"{project_name}-{env_name}-{name}"

    # 1. Network Stack (VPC, subnets, endpoints)
    network_stack = NetworkStack(
        app,
        stack_name("network"),
        env=env,
        project_name=project_name,
        env_name=env_name,
    )

    # 2. Storage Stack (S3 bucket for static content)
    storage_stack = StorageStack(
        app,
        stack_name("storage"),
        env=env,
        project_name=project_name,
        env_name=env_name,
    )

    # 3. Compute Stack (Lambda functions)
    compute_stack = ComputeStack(
        app,
        stack_name("compute"),
        env=env,
        project_name=project_name,
        env_name=env_name,
        vpc=network_stack.vpc,
        private_subnets=network_stack.private_subnets,
    )
    compute_stack.add_dependency(network_stack)

    # 4. API Stack (API Gateway with VPC Link)
    api_stack = ApiStack(
        app,
        stack_name("api"),
        env=env,
        project_name=project_name,
        env_name=env_name,
        vpc=network_stack.vpc,
        health_lambda=compute_stack.health_lambda,
        router_lambda=compute_stack.router_lambda,
    )
    api_stack.add_dependency(compute_stack)

    # 5. Load Balancer Stack (ALB + NLB)
    loadbalancer_stack = LoadBalancerStack(
        app,
        stack_name("loadbalancer"),
        env=env,
        project_name=project_name,
        env_name=env_name,
        vpc=network_stack.vpc,
        api_endpoint=api_stack.api_endpoint,
    )
    loadbalancer_stack.add_dependency(api_stack)

    # 6. CDN Stack (CloudFront distribution)
    cdn_stack = CdnStack(
        app,
        stack_name("cdn"),
        env=env,
        project_name=project_name,
        env_name=env_name,
        static_bucket=storage_stack.static_bucket,
        alb=loadbalancer_stack.alb,
        domain_name=domain_name,
    )
    cdn_stack.add_dependency(loadbalancer_stack)
    cdn_stack.add_dependency(storage_stack)

    # Apply common tags to all stacks
    for stack in [
        network_stack,
        storage_stack,
        compute_stack,
        api_stack,
        loadbalancer_stack,
        cdn_stack,
    ]:
        for key, value in common_tags.items():
            cdk.Tags.of(stack).add(key, value)

    app.synth()


if __name__ == "__main__":
    main()
