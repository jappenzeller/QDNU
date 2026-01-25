"""
Health Check Lambda Handler

Simple health check endpoint for monitoring and ALB health checks.
Returns 200 OK with service status information.
"""
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    """
    Health check handler.

    Returns service status and basic metadata.
    """
    logger.info("Health check requested")
    logger.debug(f"Event: {json.dumps(event)}")

    response_body = {
        "status": "healthy",
        "service": "qdnu-api",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "region": os.environ.get("AWS_REGION", "unknown"),
    }

    # Add Lambda context info if available
    if context:
        response_body["lambda"] = {
            "function_name": context.function_name,
            "memory_limit_mb": context.memory_limit_in_mb,
            "remaining_time_ms": context.get_remaining_time_in_millis(),
        }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
        "body": json.dumps(response_body),
    }
