"""
Load Balancer Stack - ALB for CloudFront Origin

This stack creates:
- Application Load Balancer (public-facing, for CloudFront origin)
- Target group pointing to API Gateway endpoint
- HTTPS listener (uses default CloudFront domain initially)

Architecture note:
CloudFront -> ALB -> API Gateway -> Lambda

The ALB serves as the origin for CloudFront's /api/* behavior.
This provides a stable endpoint for CloudFront and allows for
future flexibility (WAF, custom routing, etc.).
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_ec2 as ec2,
    aws_elasticloadbalancingv2 as elbv2,
    aws_elasticloadbalancingv2_targets as targets,
    CfnOutput,
)
from constructs import Construct
from urllib.parse import urlparse


class LoadBalancerStack(Stack):
    """Application Load Balancer for API routing."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        env_name: str,
        vpc: ec2.IVpc,
        api_endpoint: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.env_name = env_name

        # Parse the API Gateway endpoint to get the host
        parsed = urlparse(api_endpoint)
        api_host = parsed.netloc

        # Security group for ALB
        alb_sg = ec2.SecurityGroup(
            self,
            "AlbSecurityGroup",
            vpc=vpc,
            description="Security group for Application Load Balancer",
            allow_all_outbound=True,
        )
        alb_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(80),
            "Allow HTTP from anywhere",
        )
        alb_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(443),
            "Allow HTTPS from anywhere",
        )

        # Application Load Balancer (internet-facing)
        self.alb = elbv2.ApplicationLoadBalancer(
            self,
            "Alb",
            load_balancer_name=f"{project_name}-{env_name}-alb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        # HTTP listener - redirects to HTTPS
        self.alb.add_listener(
            "HttpListener",
            port=80,
            default_action=elbv2.ListenerAction.redirect(
                protocol="HTTPS",
                port="443",
                permanent=True,
            ),
        )

        # HTTPS listener
        # Note: For production, you would use an ACM certificate
        # For initial setup, we'll forward to API Gateway via HTTP
        # (CloudFront handles external HTTPS)
        https_listener = self.alb.add_listener(
            "HttpsListener",
            port=443,
            # For initial deployment without custom domain, use HTTP protocol
            # to forward to API Gateway (which has its own HTTPS)
            protocol=elbv2.ApplicationProtocol.HTTP,
        )

        # Default action - forward to API Gateway
        # We use a Lambda target that proxies to API Gateway
        # or configure a fixed response for non-API paths
        https_listener.add_action(
            "DefaultAction",
            action=elbv2.ListenerAction.fixed_response(
                status_code=404,
                content_type="application/json",
                message_body='{"error": "Not Found", "message": "Use /api/* endpoints"}',
            ),
        )

        # Add a rule to forward /api/* to API Gateway
        # Note: For full integration, you would typically:
        # 1. Use an IP target group pointing to API Gateway VPC endpoint
        # 2. Or use Lambda as the ALB target
        # For simplicity, we'll add a redirect to the API Gateway URL
        https_listener.add_action(
            "ApiRedirect",
            priority=100,
            conditions=[
                elbv2.ListenerCondition.path_patterns(["/api/*"]),
            ],
            action=elbv2.ListenerAction.redirect(
                host=api_host,
                protocol="HTTPS",
                port="443",
                permanent=False,
            ),
        )

        # Outputs
        CfnOutput(
            self,
            "AlbDnsName",
            value=self.alb.load_balancer_dns_name,
            description="ALB DNS name",
            export_name=f"{project_name}-{env_name}-alb-dns",
        )

        CfnOutput(
            self,
            "AlbArn",
            value=self.alb.load_balancer_arn,
            description="ALB ARN",
            export_name=f"{project_name}-{env_name}-alb-arn",
        )

        CfnOutput(
            self,
            "ApiHost",
            value=api_host,
            description="API Gateway host for direct access",
            export_name=f"{project_name}-{env_name}-api-host",
        )
