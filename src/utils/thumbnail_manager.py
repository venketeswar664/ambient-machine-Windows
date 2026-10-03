import os
import datetime
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load .env on import
load_dotenv()

class AzureDatedImageUploader:
    """
    Uploads images to Azure Blob Storage under date- and label-based directories,
    e.g. blobs like YYYY_MM_DD/label/filename.jpg

    Supports PIL Images, OpenCV ndarrays, raw bytes/bytearray, or file-like objects.
    Expects AZURE_STORAGE_CONNECTION_STRING and AZURE_CONTAINER_NAME in .env or passed in.
    """

    def __init__(
        self,
        container_name=None,
        default_container="video-storage"
    ):
        self.connection_string = (
            connection_string
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        self.container_name = (
            container_name
            or os.getenv("AZURE_CONTAINER_NAME")
            or default_container
        )
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
        if not self.container_name:
            raise ValueError("AZURE_CONTAINER_NAME is required")

        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
        try:
            if not self.container_client.exists():
                self.container_client.create_container()
        except Exception:
            pass

    def upload_image(
        self,
        image,
        image_name,
        label=None,
        timestamp=None
    ):
        """
        Uploads an image to Azure under YYYY_MM_DD/[label]/[timestamp_]image_name.

        Parameters:
        - image: PIL.Image, ndarray, bytes, or file-like
        - image_name: base filename, e.g. 'person.jpg'
        - label: optional subdirectory label, e.g. 'customer' or 'employee'
        - timestamp: optional datetime for naming; defaults to now UTC

        Returns: URL of uploaded blob.
        """
        # Determine date folder
        dt = timestamp or datetime.datetime.now(datetime.timezone.utc)
        date_str = dt.strftime("%Y_%m_%d")
        # Optionally prefix filename with timestamp
        ts_str = dt.strftime("%H%M%S")
        fname = f"{ts_str}_{image_name}" if timestamp else image_name

        # Build full blob path: date/label/filename
        parts = [date_str]
        if label:
            parts.append(label)
        parts.append(fname)
        blob_name = "/".join(parts)

        # Prepare data stream
        if isinstance(image, Image.Image):
            stream = BytesIO()
            image.save(stream, format=image.format or "PNG")
            stream.seek(0)
        elif isinstance(image, np.ndarray):
            ext = os.path.splitext(image_name)[1] or ".png"
            ok, enc = cv2.imencode(ext, image)
            if not ok:
                raise IOError("Failed to encode ndarray")
            stream = BytesIO(enc.tobytes())
        elif isinstance(image, (bytes, bytearray)):
            stream = BytesIO(image)
        elif hasattr(image, 'read'):
            stream = BytesIO(image.read())
        else:
            raise TypeError("Unsupported image type")

        client = self.container_client.get_blob_client(blob_name)
        client.upload_blob(stream, overwrite=True)

        acct = self.blob_service_client.account_name
        url = f"https://{acct}.blob.core.windows.net/{self.container_name}/{blob_name}"
        print(f"✅ Uploaded: {url}")
        return url

