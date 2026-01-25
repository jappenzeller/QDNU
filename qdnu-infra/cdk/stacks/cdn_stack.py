"""
CDN Stack - CloudFront Distribution

This stack creates:
- CloudFront distribution as single entry point
- S3 origin for static content (default behavior)
- ALB origin for API requests (/api/*)
- Origin Access Control for secure S3 access
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_s3 as s3,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    CfnOutput,
)
from constructs import Construct


class CdnStack(Stack):
    """CloudFront distribution for QDNU application."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        env_name: str,
        static_bucket: s3.IBucket,
        alb: elbv2.IApplicationLoadBalancer,
        domain_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.env_name = env_name

        # Origin Access Control for S3
        oac = cloudfront.S3OriginAccessControl(
            self,
            "OAC",
            description=f"OAC for {project_name}-{env_name} static content",
        )

        # S3 origin for static content
        s3_origin = origins.S3BucketOrigin.with_origin_access_control(
            static_bucket,
            origin_access_control=oac,
        )

        # ALB origin for API
        alb_origin = origins.HttpOrigin(
            alb.load_balancer_dns_name,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
            http_port=80,
            origin_id="api-origin",
        )

        # Cache policy for static assets
        static_cache_policy = cloudfront.CachePolicy(
            self,
            "StaticCachePolicy",
            cache_policy_name=f"{project_name}-{env_name}-static-cache",
            comment="Cache policy for static assets",
            default_ttl=Duration.days(1),
            max_ttl=Duration.days(365),
            min_ttl=Duration.seconds(0),
            cookie_behavior=cloudfront.CacheCookieBehavior.none(),
            header_behavior=cloudfront.CacheHeaderBehavior.none(),
            query_string_behavior=cloudfront.CacheQueryStringBehavior.none(),
            enable_accept_encoding_gzip=True,
            enable_accept_encoding_brotli=True,
        )

        # Cache policy for API (no caching)
        api_cache_policy = cloudfront.CachePolicy(
            self,
            "ApiCachePolicy",
            cache_policy_name=f"{project_name}-{env_name}-api-cache",
            comment="No-cache policy for API requests",
            default_ttl=Duration.seconds(0),
            max_ttl=Duration.seconds(0),
            min_ttl=Duration.seconds(0),
            cookie_behavior=cloudfront.CacheCookieBehavior.all(),
            header_behavior=cloudfront.CacheHeaderBehavior.allow_list(
                "Authorization",
                "Content-Type",
                "Origin",
                "Accept",
            ),
            query_string_behavior=cloudfront.CacheQueryStringBehavior.all(),
        )

        # Origin request policy for API (forward all)
        api_origin_request_policy = cloudfront.OriginRequestPolicy(
            self,
            "ApiOriginRequestPolicy",
            origin_request_policy_name=f"{project_name}-{env_name}-api-origin-request",
            comment="Forward all headers, cookies, and query strings to API",
            cookie_behavior=cloudfront.OriginRequestCookieBehavior.all(),
            header_behavior=cloudfront.OriginRequestHeaderBehavior.allow_list(
                "Accept",
                "Accept-Language",
                "Content-Type",
                "Origin",
                "Referer",
            ),
            query_string_behavior=cloudfront.OriginRequestQueryStringBehavior.all(),
        )

        # Response headers policy (security headers)
        response_headers_policy = cloudfront.ResponseHeadersPolicy(
            self,
            "SecurityHeadersPolicy",
            response_headers_policy_name=f"{project_name}-{env_name}-security-headers",
            comment="Security headers for all responses",
            security_headers_behavior=cloudfront.ResponseSecurityHeadersBehavior(
                content_type_options=cloudfront.ResponseHeadersContentTypeOptions(
                    override=True
                ),
                frame_options=cloudfront.ResponseHeadersFrameOptions(
                    frame_option=cloudfront.HeadersFrameOption.DENY,
                    override=True,
                ),
                referrer_policy=cloudfront.ResponseHeadersReferrerPolicy(
                    referrer_policy=cloudfront.HeadersReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN,
                    override=True,
                ),
                strict_transport_security=cloudfront.ResponseHeadersStrictTransportSecurity(
                    access_control_max_age=Duration.days(365),
                    include_subdomains=True,
                    override=True,
                ),
                xss_protection=cloudfront.ResponseHeadersXSSProtection(
                    protection=True,
                    mode_block=True,
                    override=True,
                ),
            ),
        )

        # CloudFront distribution
        self.distribution = cloudfront.Distribution(
            self,
            "Distribution",
            comment=f"QDNU {env_name} distribution",
            default_root_object="index.html",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,  # US, Canada, Europe
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
            minimum_protocol_version=cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
            # Default behavior: S3 static content
            default_behavior=cloudfront.BehaviorOptions(
                origin=s3_origin,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                cached_methods=cloudfront.CachedMethods.CACHE_GET_HEAD_OPTIONS,
                cache_policy=static_cache_policy,
                response_headers_policy=response_headers_policy,
                compress=True,
            ),
            # Additional behaviors
            additional_behaviors={
                # API behavior: /api/*
                "/api/*": cloudfront.BehaviorOptions(
                    origin=alb_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cached_methods=cloudfront.CachedMethods.CACHE_GET_HEAD_OPTIONS,
                    cache_policy=api_cache_policy,
                    origin_request_policy=api_origin_request_policy,
                    response_headers_policy=response_headers_policy,
                    compress=True,
                ),
            },
            # Error pages - serve index.html for SPA routing
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
            ],
        )

        # Grant CloudFront access to S3 bucket
        static_bucket.add_to_resource_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[static_bucket.arn_for_objects("*")],
                principals=[iam.ServicePrincipal("cloudfront.amazonaws.com")],
                conditions={
                    "StringEquals": {
                        "AWS:SourceArn": f"arn:aws:cloudfront::{self.account}:distribution/{self.distribution.distribution_id}"
                    }
                },
            )
        )

        # Outputs
        CfnOutput(
            self,
            "DistributionId",
            value=self.distribution.distribution_id,
            description="CloudFront distribution ID",
            export_name=f"{project_name}-{env_name}-distribution-id",
        )

        CfnOutput(
            self,
            "DistributionDomainName",
            value=self.distribution.distribution_domain_name,
            description="CloudFront distribution domain name",
            export_name=f"{project_name}-{env_name}-distribution-domain",
        )

        CfnOutput(
            self,
            "WebsiteUrl",
            value=f"https://{self.distribution.distribution_domain_name}",
            description="Website URL",
            export_name=f"{project_name}-{env_name}-website-url",
        )
