"""MedImageInsight-Premium embedding client and image preparation (research use, not diagnosis).

The Premium deployment exposes one REST call:

    POST https://<resource>.services.ai.azure.com/providers/microsoft/v2/embed

Text and images go in **separate requests** and come back as 1024-dim float vectors.
`PremiumClient.embed` unit-normalises them, so a dot product is already a cosine similarity.
"""

import base64
import io
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image, ImageOps

EMBED_PATH = "/providers/microsoft/v2/embed"
TOKEN_SCOPE = "https://ai.azure.com/.default"
DIMENSIONS = 1024
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".svs"}
MAX_SVS_LEVEL_PIXELS = 16_000_000


def resource_endpoint(value):
    """Accept a project URL, a resource URL, or an already-complete embed URL."""
    parts = urlsplit(value.strip())
    if (parts.scheme != "https" or not parts.netloc or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        raise ValueError("Use an HTTPS endpoint without credentials, query, or fragment.")
    path = parts.path.rstrip("/")
    if path.startswith("/api/projects/"):
        path = ""
    if not path.endswith(EMBED_PATH):
        path += EMBED_PATH
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def ensure_azure_cli_path():
    """Find the standard Windows install of `az` without touching the persistent PATH."""
    if shutil.which("az") or os.name != "nt":
        return
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(variable)
        folder = Path(base) / "Microsoft SDKs" / "Azure" / "CLI2" / "wbin" if base else None
        if folder and (folder / "az.cmd").is_file():
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError("Azure CLI not found. Install it, or add its wbin directory to PATH.")


def load_labels(path):
    """Read a {label: english prompt} JSON file."""
    labels = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(labels, dict) or len(labels) < 2:
        raise ValueError("Labels must be a JSON object with at least two label/prompt pairs.")
    if any(not isinstance(key, str) or not key.strip()
           or not isinstance(value, str) or not value.strip() for key, value in labels.items()):
        raise ValueError("Label names and prompts must be nonempty strings.")
    if len({value.strip() for value in labels.values()}) != len(labels):
        raise ValueError("Each label must have a different prompt.")
    return labels


def normalize_vectors(values, count):
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.shape != (count, DIMENSIONS):
        raise ValueError(f"Expected embeddings shape {(count, DIMENSIONS)}, got {vectors.shape}.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Embeddings contain a zero or invalid norm.")
    return vectors / norms


class PremiumClient:
    """One HTTP session against the Premium embed endpoint. Vectors come back unit-normalised."""

    def __init__(self, endpoint, model, credential):
        self.endpoint = endpoint
        self.model = model
        self.credential = credential
        self.session = requests.Session()
        self.calls = []

    def embed(self, modality, inputs):
        """modality is "texts" or "images"; one modality per request. Returns [len(inputs), 1024]."""
        if modality not in ("texts", "images") or not inputs:
            raise ValueError("Use one nonempty texts OR images list per request.")
        token = self.credential.get_token(TOKEN_SCOPE)
        # Never log the bearer token, the payload, or the raw response body.
        response = self.session.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {token.token}"},
            json={"model": self.model, modality: inputs, "embedding_types": ["float"]},
            timeout=(15, 120),
            allow_redirects=False,
        )
        request_id = response.headers.get("apim-request-id") or response.headers.get("x-request-id")
        if response.status_code != 200:
            raise RuntimeError(
                f"Premium API HTTP {response.status_code}; request_id={request_id}. "
                "Check the deployment endpoint/name, tenant, RBAC, and quota. "
                "No automatic retries were made."
            )
        body = response.json()
        vectors = normalize_vectors(body["embeddings"]["float"], len(inputs))
        self.calls.append({"modality": modality, "items": len(inputs),
                           "request_id": request_id, "response_id": body.get("id")})
        return vectors

    def close(self):
        self.session.close()
        self.credential.close()


def png_data_url(image):
    """PIL image -> data URL, re-encoded so no source metadata rides along."""
    with io.BytesIO() as stream:
        clean = Image.new("RGB", image.size)
        clean.paste(image.convert("RGB"))
        clean.save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def file_data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def svs_overview(path):
    """Read a bounded RGB pyramid level - a slide overview, never the full-resolution WSI."""
    import tifffile

    with tifffile.TiffFile(path) as slide:
        # series[0] is the tissue pyramid; the label and macro images are separate series.
        series = slide.series[0]
        candidates = [(index, level) for index, level in enumerate(series.levels)
                      if level.axes == "YXS" and len(level.shape) == 3
                      and level.shape[2] in (3, 4) and level.dtype == np.dtype("uint8")
                      and level.shape[0] * level.shape[1] <= MAX_SVS_LEVEL_PIXELS]
        if not candidates:
            raise ValueError(
                f"{Path(path).name}: no safe RGB pyramid level (<= {MAX_SVS_LEVEL_PIXELS} pixels). "
                "Do not decode the full slide here; tile it on the cluster instead."
            )
        index, level = min(candidates, key=lambda item: item[1].shape[0] * item[1].shape[1])
        image = Image.fromarray(level.asarray())
        metadata = {
            "method": "svs_lowest_resolution_tissue_pyramid_overview",
            "original_size": [int(series.shape[1]), int(series.shape[0])],
            "pyramid_level": index,
            "pyramid_size": [int(level.shape[1]), int(level.shape[0])],
            "limitation": "Downsampled whole-slide overview, not a cellular-level evaluation.",
        }
    return image, metadata


def prepare_image(path, destination, max_side):
    """Convert one PNG/JPEG/SVS into the RGB PNG that will actually be sent."""
    path = Path(path)
    if max_side < 1:
        raise ValueError("max_side must be positive.")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {suffix}")
    if suffix == ".svs":
        image, metadata = svs_overview(path)
    else:
        with Image.open(path) as source:
            if source.format not in ("PNG", "JPEG"):
                raise ValueError(f"{path.name}: expected a PNG or JPEG file.")
            image = ImageOps.exif_transpose(source).copy()
        metadata = {"method": "rgb_png", "original_size": list(image.size)}

    rgba = image.convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    clean.save(destination, format="PNG")
    metadata.update({"prepared_size": list(clean.size), "prepared_image": str(destination)})
    return metadata


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
