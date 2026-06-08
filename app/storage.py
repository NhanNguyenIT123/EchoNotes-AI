from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    def put_file(self, local_path: Path, object_name: str) -> str:
        ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, local_path: Path, object_name: str) -> str:
        target = self.root / object_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)
        return str(target)


def get_storage_backend() -> StorageBackend:
    provider = os.getenv("ECHONOTES_STORAGE_PROVIDER", "local").lower()
    if provider == "azure":
        # Lazy import keeps local-first setup dependency-light.
        from azure.storage.blob import BlobServiceClient  # type: ignore

        conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        container = os.getenv("AZURE_STORAGE_CONTAINER", "echonotes")
        service = BlobServiceClient.from_connection_string(conn)
        client = service.get_container_client(container)
        try:
            client.create_container()
        except Exception:
            pass

        class AzureBlobStorage:
            def put_file(self, local_path: Path, object_name: str) -> str:
                blob = client.get_blob_client(object_name)
                with Path(local_path).open("rb") as f:
                    blob.upload_blob(f, overwrite=True)
                return blob.url

        return AzureBlobStorage()

    if provider == "s3":
        import boto3  # type: ignore

        bucket = os.environ["ECHONOTES_S3_BUCKET"]
        s3 = boto3.client("s3")

        class S3Storage:
            def put_file(self, local_path: Path, object_name: str) -> str:
                s3.upload_file(str(local_path), bucket, object_name)
                region = s3.meta.region_name or "us-east-1"
                if region == "us-east-1":
                    return f"https://{bucket}.s3.amazonaws.com/{object_name}"
                return f"https://{bucket}.s3.{region}.amazonaws.com/{object_name}"

        return S3Storage()

    return LocalStorage(Path(os.getenv("ECHONOTES_LOCAL_STORAGE_DIR", "data/cloud_artifacts")))
