import boto3
from botocore.exceptions import ClientError
from config import settings


def _get_client():
    kwargs = {
        "region_name": settings.aws_s3_region,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
    }
    if settings.aws_session_token:
        kwargs["aws_session_token"] = settings.aws_session_token
    return boto3.client("s3", **kwargs)


def upload_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    client = _get_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def get_presigned_url(bucket: str, key: str, expires_in: int = 3600) -> str:
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
