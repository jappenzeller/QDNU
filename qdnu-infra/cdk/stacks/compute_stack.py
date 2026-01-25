"""
Compute Stack - Lambda Functions

This stack creates:
- Health check Lambda function
- Router Lambda function (handles /api/{proxy+} requests)
- Lambda functions deployed in VPC private subnets
"""
import os
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as lambda_,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_logs as logs,
    CfnOutput,
)
from constructs import Construct


class ComputeStack(Stack):
    """Lambda functions for API backend."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        env_name: str,
        vpc: ec2.IVpc,
        private_subnets: ec2.SelectedSubnets,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.env_name = env_name

        # Lambda security group
        self.lambda_sg = ec2.SecurityGroup(
            self,
            "LambdaSecurityGroup",
            vpc=vpc,
            description="Security group for Lambda functions",
            allow_all_outbound=True,
        )

        # Common Lambda configuration
        lambda_runtime = lambda_.Runtime.PYTHON_3_11
        lambda_code_path = os.path.join(os.path.dirname(__file__), "..", "..", "lambda")

        # Common environment variables
        common_env = {
            "PROJECT_NAME": project_name,
            "ENVIRONMENT": env_name,
            "LOG_LEVEL": "DEBUG" if env_name == "dev" else "INFO",
        }

        # IAM role for Lambda functions
        lambda_role = iam.Role(
            self,
            "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaVPCAccessExecutionRole"
                ),
            ],
        )

        # Health check Lambda
        self.health_lambda = lambda_.Function(
            self,
            "HealthLambda",
            function_name=f"{project_name}-{env_name}-health",
            runtime=lambda_runtime,
            handler="handler.handler",
            code=lambda_.Code.from_asset(os.path.join(lambda_code_path, "health")),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[self.lambda_sg],
            role=lambda_role,
            timeout=Duration.seconds(10),
            memory_size=128,
            environment=common_env,
            log_retention=logs.RetentionDays.ONE_WEEK if env_name == "dev" else logs.RetentionDays.ONE_MONTH,
        )

        # Router Lambda (handles /api/{proxy+})
        self.router_lambda = lambda_.Function(
            self,
            "RouterLambda",
            function_name=f"{project_name}-{env_name}-router",
            runtime=lambda_runtime,
            handler="handler.handler",
            code=lambda_.Code.from_asset(os.path.join(lambda_code_path, "router")),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[self.lambda_sg],
            role=lambda_role,
            timeout=Duration.seconds(30),
            memory_size=256,
            environment=common_env,
            log_retention=logs.RetentionDays.ONE_WEEK if env_name == "dev" else logs.RetentionDays.ONE_MONTH,
        )

        # Outputs
        CfnOutput(
            self,
            "HealthLambdaArn",
            value=self.health_lambda.function_arn,
            description="Health Lambda function ARN",
            export_name=f"{project_name}-{env_name}-health-lambda-arn",
        )

        CfnOutput(
            self,
            "RouterLambdaArn",
            value=self.router_lambda.function_arn,
            description="Router Lambda function ARN",
            export_name=f"{project_name}-{env_name}-router-lambda-arn",
        )
