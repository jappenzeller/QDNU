# QDNU Infrastructure

AWS CDK infrastructure for the QDNU (Quantum Positive-Negative Neuron) web application.

## Architecture

```
CloudFront (single entry point)
├── /api/* → ALB → API Gateway → Lambda functions
└── /* → S3 bucket (static content)
```

### Components

| Stack | Description |
|-------|-------------|
| NetworkStack | VPC, subnets, VPC endpoints |
| StorageStack | S3 bucket for static content |
| ComputeStack | Lambda functions (health, router) |
| ApiStack | HTTP API Gateway with Lambda integrations |
| LoadBalancerStack | Application Load Balancer |
| CdnStack | CloudFront distribution |

## Prerequisites

- Python 3.11+
- AWS CLI configured with credentials
- Node.js 18+ (for CDK CLI)
- AWS CDK CLI: `npm install -g aws-cdk`

## Setup

1. **Clone and navigate to the infrastructure directory:**

   ```bash
   cd qdnu-infra
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**

   ```bash
   cd cdk
   pip install -r requirements.txt
   ```

4. **Bootstrap CDK (first time only):**

   ```bash
   cdk bootstrap aws://ACCOUNT-ID/REGION
   ```

## Deployment

### Deploy all stacks

```bash
cd cdk
cdk deploy --all
```

### Deploy individual stacks

```bash
cdk deploy qdnu-dev-network
cdk deploy qdnu-dev-storage
cdk deploy qdnu-dev-compute
cdk deploy qdnu-dev-api
cdk deploy qdnu-dev-loadbalancer
cdk deploy qdnu-dev-cdn
```

### Deploy to production

```bash
cdk deploy --all --context environment=prod
```

## Configuration

Configuration is managed through CDK context in `cdk.json`:

| Key | Description | Default |
|-----|-------------|---------|
| `environment` | Deployment environment (dev/prod) | dev |
| `project_name` | Project name prefix | qdnu |
| `domain_name` | Custom domain (optional) | "" |

Override at deploy time:

```bash
cdk deploy --all --context environment=prod --context domain_name=qdnu.example.com
```

## Upload Static Content

After deployment, upload your static files to S3:

```bash
aws s3 sync ./static s3://qdnu-dev-static-ACCOUNT_ID/ --delete
```

## Useful Commands

| Command | Description |
|---------|-------------|
| `cdk ls` | List all stacks |
| `cdk synth` | Synthesize CloudFormation template |
| `cdk diff` | Compare deployed stack with current state |
| `cdk deploy` | Deploy stack(s) to AWS |
| `cdk destroy` | Destroy stack(s) |

## API Endpoints

After deployment, the following endpoints are available:

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/users` | List users |
| `GET /api/users/{id}` | Get user by ID |
| `POST /api/users` | Create user |
| `GET /api/data` | Get QDNU data |
| `POST /api/analyze` | Submit analysis |

## Outputs

After deployment, CDK outputs useful values:

- **WebsiteUrl**: CloudFront distribution URL
- **ApiEndpoint**: Direct API Gateway URL
- **StaticBucketName**: S3 bucket for static files
- **DistributionId**: CloudFront distribution ID

## Cost Considerations

- **VPC Endpoints**: ~$7.20/month per endpoint (3 endpoints = ~$22/month)
- **ALB**: ~$16/month base + data processing
- **CloudFront**: Pay per request + data transfer
- **Lambda**: Pay per invocation (free tier: 1M requests/month)
- **S3**: Pay per storage + requests (minimal for static sites)

For development, consider:
- Reducing VPC endpoints (remove Lambda endpoint if not needed)
- Using smaller instance types

## Future Additions

- [ ] Cognito User Pool with Amazon IdP
- [ ] Lambda authorizer for API Gateway
- [ ] Custom domain with Route 53
- [ ] WAF rules on CloudFront
- [ ] CI/CD pipeline with CodePipeline

## Troubleshooting

### Lambda timeout in VPC

If Lambda functions timeout, ensure:
1. VPC endpoints are configured correctly
2. Security groups allow outbound traffic
3. Lambda has correct IAM permissions

### CloudFront 403 errors

If S3 returns 403:
1. Check OAC is configured correctly
2. Verify S3 bucket policy includes CloudFront principal
3. Ensure files exist in the bucket

### API Gateway 5xx errors

Check CloudWatch Logs for Lambda function errors:

```bash
aws logs tail /aws/lambda/qdnu-dev-health --follow
aws logs tail /aws/lambda/qdnu-dev-router --follow
```
