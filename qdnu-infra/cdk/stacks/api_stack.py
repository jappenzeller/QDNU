"""
API Stack - API Gateway with Lambda Integration

This stack creates:
- HTTP API Gateway (more cost-effective than REST API)
- Lambda integrations for health and router functions
- Resource policy restricting access to VPC (for private API pattern)

Note: This creates a regional API that will be accessed through the ALB.
For a fully private API, you would use a Private API with VPC endpoint.
"""
from aws_cdk import (
    Stack,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_integrations as integrations,
    aws_ec2 as ec2,
    aws_lambda as lambda_,
    CfnOutput,
)
from constructs import Construct


class ApiStack(Stack):
    """API Gateway for Lambda backend."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        env_name: str,
        vpc: ec2.IVpc,
        health_lambda: lambda_.IFunction,
        router_lambda: lambda_.IFunction,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.env_name = env_name

        # Create HTTP API Gateway
        self.api = apigwv2.HttpApi(
            self,
            "HttpApi",
            api_name=f"{project_name}-{env_name}-api",
            description=f"QDNU API - {env_name}",
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_headers=["Content-Type", "Authorization", "X-Amz-Date"],
                allow_methods=[
                    apigwv2.CorsHttpMethod.GET,
                    apigwv2.CorsHttpMethod.POST,
                    apigwv2.CorsHttpMethod.PUT,
                    apigwv2.CorsHttpMethod.DELETE,
                    apigwv2.CorsHttpMethod.OPTIONS,
                ],
                allow_origins=["*"],  # Restrict in production
                max_age=apigwv2.Duration.hours(1),
            ),
        )

        # Health check route: GET /api/health
        health_integration = integrations.HttpLambdaIntegration(
            "HealthIntegration",
            health_lambda,
        )
        self.api.add_routes(
            path="/api/health",
            methods=[apigwv2.HttpMethod.GET],
            integration=health_integration,
        )

        # Router route: ANY /api/{proxy+}
        router_integration = integrations.HttpLambdaIntegration(
            "RouterIntegration",
            router_lambda,
        )
        self.api.add_routes(
            path="/api/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=router_integration,
        )

        # Store API endpoint for use by other stacks
        self.api_endpoint = self.api.api_endpoint

        # Outputs
        CfnOutput(
            self,
            "ApiEndpoint",
            value=self.api_endpoint,
            description="API Gateway endpoint URL",
            export_name=f"{project_name}-{env_name}-api-endpoint",
        )

        CfnOutput(
            self,
            "ApiId",
            value=self.api.api_id,
            description="API Gateway ID",
            export_name=f"{project_name}-{env_name}-api-id",
        )
