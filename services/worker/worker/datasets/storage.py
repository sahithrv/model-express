from __future__ import annotations
import boto3
import os
from pathlib import Path
from urllib.parse import urlparse

def download_s3_uri(storage_uri: str, destination: Path) -> Path:
    bucket, key = parse_s3_uri(storage_uri)

    s3client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "model_express"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "model_express_password"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    s3client.download_file(bucket, key, str(destination))
    return destination

def parse_s3_uri(storage_uri: str) -> tuple[str, str]:
    parsed = urlparse(storage_uri)

    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {storage_uri}")
    
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {storage_uri}")

    return bucket, key
