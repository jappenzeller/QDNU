"""
Router Lambda Handler

Main API router that handles all /api/{proxy+} requests.
Implements the "embedded Lambda" pattern - single function routing
to multiple logical endpoints.
"""
import json
import os
import logging
import re
from typing import Any, Callable
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


# Route registry
ROUTES: dict[str, dict[str, Callable]] = {}


def route(method: str, pattern: str):
    """Decorator to register a route handler."""
    def decorator(func: Callable):
        if pattern not in ROUTES:
            ROUTES[pattern] = {}
        ROUTES[pattern][method.upper()] = func
        return func
    return decorator


def match_route(path: str, method: str) -> tuple[Callable | None, dict]:
    """Match a path against registered routes."""
    for pattern, methods in ROUTES.items():
        # Convert route pattern to regex
        # /api/users/{id} -> /api/users/(?P<id>[^/]+)
        regex_pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', pattern)
        regex_pattern = f"^{regex_pattern}$"

        match = re.match(regex_pattern, path)
        if match and method in methods:
            return methods[method], match.groupdict()

    return None, {}


# =============================================================================
# ROUTE HANDLERS
# =============================================================================

@route("GET", "/api/users")
def list_users(event: dict, context: Any, params: dict) -> dict:
    """List all users (placeholder)."""
    return {
        "users": [
            {"id": "1", "name": "Alice", "email": "alice@example.com"},
            {"id": "2", "name": "Bob", "email": "bob@example.com"},
        ],
        "total": 2,
    }


@route("GET", "/api/users/{id}")
def get_user(event: dict, context: Any, params: dict) -> dict:
    """Get user by ID (placeholder)."""
    user_id = params.get("id")
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
    }


@route("POST", "/api/users")
def create_user(event: dict, context: Any, params: dict) -> dict:
    """Create a new user (placeholder)."""
    body = json.loads(event.get("body") or "{}")
    return {
        "id": "new-user-id",
        "name": body.get("name", "New User"),
        "email": body.get("email", "new@example.com"),
        "created": True,
    }


@route("GET", "/api/data")
def get_data(event: dict, context: Any, params: dict) -> dict:
    """Get data endpoint (placeholder for QDNU data)."""
    return {
        "message": "QDNU data endpoint",
        "data": {
            "quantum_states": [],
            "predictions": [],
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@route("POST", "/api/analyze")
def analyze(event: dict, context: Any, params: dict) -> dict:
    """Analyze endpoint (placeholder for QDNU analysis)."""
    body = json.loads(event.get("body") or "{}")
    return {
        "message": "Analysis request received",
        "input": body,
        "result": {
            "status": "processing",
            "job_id": "analysis-job-123",
        },
    }


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handler(event, context):
    """
    Main router handler.

    Routes requests to appropriate handlers based on path and method.
    """
    logger.info("Router invoked")
    logger.debug(f"Event: {json.dumps(event)}")

    # Extract request info
    request_context = event.get("requestContext", {})
    http = request_context.get("http", {})

    path = http.get("path", event.get("rawPath", "/"))
    method = http.get("method", event.get("httpMethod", "GET"))

    logger.info(f"Request: {method} {path}")

    # Match route
    handler_func, params = match_route(path, method)

    if handler_func:
        try:
            result = handler_func(event, context, params)
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "X-Request-Id": request_context.get("requestId", "unknown"),
                },
                "body": json.dumps(result),
            }
        except Exception as e:
            logger.exception(f"Handler error: {e}")
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "Internal Server Error",
                    "message": str(e) if os.environ.get("ENVIRONMENT") == "dev" else "An error occurred",
                }),
            }

    # No matching route
    return {
        "statusCode": 404,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "error": "Not Found",
            "message": f"No handler for {method} {path}",
            "available_routes": list(ROUTES.keys()),
        }),
    }
