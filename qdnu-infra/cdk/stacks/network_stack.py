"""
Network Stack - VPC, Subnets, and VPC Endpoints

This stack creates the foundational network infrastructure:
- VPC with private subnets (no public subnets needed for this architecture)
- VPC endpoints for AWS services (API Gateway execute-api)
- Security groups for internal communication
"""
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    CfnOutput,
)
from constructs import Construct


class NetworkStack(Stack):
    """VPC and network infrastructure for QDNU."""

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

        # Create VPC with private and public subnets
        # Public subnets needed for ALB, private for NLB and Lambda
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            vpc_name=f"{project_name}-{env_name}-vpc",
            max_azs=2,
            nat_gateways=0,  # No NAT gateway initially (cost saving)
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
        )

        # Store subnet selections for easy access
        self.public_subnets = self.vpc.select_subnets(
            subnet_type=ec2.SubnetType.PUBLIC
        )
        self.private_subnets = self.vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
        )

        # Security group for VPC endpoints
        self.endpoint_sg = ec2.SecurityGroup(
            self,
            "EndpointSecurityGroup",
            vpc=self.vpc,
            description="Security group for VPC endpoints",
            allow_all_outbound=True,
        )
        self.endpoint_sg.add_ingress_rule(
            ec2.Peer.ipv4(self.vpc.vpc_cidr_block),
            ec2.Port.tcp(443),
            "Allow HTTPS from VPC",
        )

        # Security group for Lambda functions
        self.lambda_sg = ec2.SecurityGroup(
            self,
            "LambdaSecurityGroup",
            vpc=self.vpc,
            description="Security group for Lambda functions",
            allow_all_outbound=True,
        )

        # Security group for NLB
        self.nlb_sg = ec2.SecurityGroup(
            self,
            "NlbSecurityGroup",
            vpc=self.vpc,
            description="Security group for Network Load Balancer",
            allow_all_outbound=True,
        )
        self.nlb_sg.add_ingress_rule(
            ec2.Peer.ipv4(self.vpc.vpc_cidr_block),
            ec2.Port.tcp(443),
            "Allow HTTPS from VPC",
        )

        # VPC Endpoint for API Gateway (execute-api)
        self.api_gateway_endpoint = ec2.InterfaceVpcEndpoint(
            self,
            "ApiGatewayEndpoint",
            vpc=self.vpc,
            service=ec2.InterfaceVpcEndpointAwsService.APIGATEWAY,
            subnets=self.private_subnets,
            security_groups=[self.endpoint_sg],
            private_dns_enabled=True,
        )

        # VPC Endpoint for Lambda (allows Lambda in VPC to call AWS services)
        self.lambda_endpoint = ec2.InterfaceVpcEndpoint(
            self,
            "LambdaEndpoint",
            vpc=self.vpc,
            service=ec2.InterfaceVpcEndpointAwsService.LAMBDA_,
            subnets=self.private_subnets,
            security_groups=[self.endpoint_sg],
            private_dns_enabled=True,
        )

        # VPC Endpoint for CloudWatch Logs (Lambda logging)
        self.logs_endpoint = ec2.InterfaceVpcEndpoint(
            self,
            "LogsEndpoint",
            vpc=self.vpc,
            service=ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
            subnets=self.private_subnets,
            security_groups=[self.endpoint_sg],
            private_dns_enabled=True,
        )

        # Outputs
        CfnOutput(
            self,
            "VpcId",
            value=self.vpc.vpc_id,
            description="VPC ID",
            export_name=f"{project_name}-{env_name}-vpc-id",
        )

        CfnOutput(
            self,
            "PrivateSubnetIds",
            value=",".join([s.subnet_id for s in self.private_subnets.subnets]),
            description="Private subnet IDs",
            export_name=f"{project_name}-{env_name}-private-subnet-ids",
        )

        CfnOutput(
            self,
            "PublicSubnetIds",
            value=",".join([s.subnet_id for s in self.public_subnets.subnets]),
            description="Public subnet IDs",
            export_name=f"{project_name}-{env_name}-public-subnet-ids",
        )
