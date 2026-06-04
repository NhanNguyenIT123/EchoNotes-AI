import base64
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests


GRAPH_API_ROOT = "https://graph.microsoft.com/v1.0"


def get_default_graph_config() -> Dict[str, str]:
    return {
        "tenant_id": os.getenv("MICROSOFT_TENANT_ID", "common"),
        "client_id": os.getenv("MICROSOFT_CLIENT_ID", ""),
        "scopes": os.getenv("MICROSOFT_GRAPH_SCOPES", "User.Read Files.Read.All"),
    }


def encode_sharing_url(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"u!{encoded}"


def create_device_flow(client_id: str, tenant_id: str, scopes: List[str]) -> Dict:
    try:
        import msal
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'msal'. Run setup_and_run.bat to install dependencies.") from exc

    authority = f"https://login.microsoftonline.com/{tenant_id or 'common'}"
    app = msal.PublicClientApplication(client_id=client_id, authority=authority)
    flow = app.initiate_device_flow(scopes=scopes)
    if "user_code" not in flow:
        raise RuntimeError(f"Could not create Microsoft device login flow: {flow}")

    flow["_client_id"] = client_id
    flow["_tenant_id"] = tenant_id
    flow["_scopes"] = scopes
    return flow


def complete_device_flow(flow: Dict) -> Dict:
    try:
        import msal
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'msal'. Run setup_and_run.bat to install dependencies.") from exc

    client_id = flow.get("_client_id")
    tenant_id = flow.get("_tenant_id") or "common"
    if not client_id:
        raise RuntimeError("Device flow is missing client id. Start Microsoft login again.")

    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = msal.PublicClientApplication(client_id=client_id, authority=authority)
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        error = result.get("error_description") or result.get("error") or "Unknown Microsoft login error."
        raise RuntimeError(error)

    result["device_flow_message"] = flow.get("message", "")
    return result


def acquire_device_token(client_id: str, tenant_id: str, scopes: List[str]) -> Dict:
    flow = create_device_flow(client_id, tenant_id, scopes)
    return complete_device_flow(flow)


def graph_get(access_token: str, url: str) -> Dict:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Graph API error {response.status_code}: {response.text[:1000]}")
    return response.json()


def resolve_share_link(access_token: str, sharing_url: str) -> Dict:
    share_id = encode_sharing_url(sharing_url)
    return graph_get(access_token, f"{GRAPH_API_ROOT}/shares/{share_id}/driveItem")


def resolve_sharepoint_stream_link(access_token: str, stream_url: str) -> Dict:
    """
    Resolve Teams/Stream SharePoint player URLs such as:
      https://tenant.sharepoint.com/sites/Site/_layouts/15/stream.aspx?id=/sites/Site/Shared Documents/...

    Graph /shares often rejects player URLs, so this extracts the document path
    and resolves the file through the site's default drive.
    """
    parsed = urlparse(stream_url)
    host = parsed.netloc
    query = parse_qs(parsed.query)
    raw_id = (query.get("id") or [""])[0]
    if not host or not raw_id:
        raise RuntimeError("This is not a supported SharePoint stream.aspx URL.")

    file_path = unquote(raw_id)
    parts = [part for part in file_path.split("/") if part]
    if len(parts) < 4 or parts[0].lower() != "sites":
        raise RuntimeError(f"Unsupported SharePoint file path in URL: {file_path}")

    site_name = parts[1]
    library_index = None
    for idx in range(2, len(parts)):
        if parts[idx].lower() in {"shared documents", "documents"}:
            library_index = idx
            break
    if library_index is None:
        raise RuntimeError(f"Could not identify SharePoint document library in path: {file_path}")

    relative_file_path = "/".join(parts[library_index + 1:])
    if not relative_file_path:
        raise RuntimeError("Could not identify recording file path in SharePoint URL.")

    site = graph_get(access_token, f"{GRAPH_API_ROOT}/sites/{host}:/sites/{site_name}")
    site_id = site.get("id")
    if not site_id:
        raise RuntimeError("Graph did not return a SharePoint site id.")

    return graph_get(
        access_token,
        f"{GRAPH_API_ROOT}/sites/{site_id}/drive/root:/{relative_file_path}"
    )


def resolve_recording_link(access_token: str, recording_url: str) -> Dict:
    errors = []
    try:
        return resolve_share_link(access_token, recording_url)
    except Exception as exc:
        errors.append(f"/shares failed: {exc}")

    try:
        return resolve_sharepoint_stream_link(access_token, recording_url)
    except Exception as exc:
        errors.append(f"stream.aspx path fallback failed: {exc}")

    raise RuntimeError("Could not resolve recording link. " + " | ".join(errors))


def download_drive_item(access_token: str, item: Dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = item.get("name") or "teams_recording.mp4"
    output_path = output_dir / name

    download_url = item.get("@microsoft.graph.downloadUrl")
    if download_url:
        response = requests.get(download_url, timeout=300)
    else:
        drive_id = item.get("parentReference", {}).get("driveId")
        item_id = item.get("id")
        if not drive_id or not item_id:
            raise RuntimeError("Graph item is missing driveId/itemId; cannot download.")
        response = requests.get(
            f"{GRAPH_API_ROOT}/drives/{drive_id}/items/{item_id}/content",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=300,
            allow_redirects=True,
        )

    if response.status_code >= 400:
        raise RuntimeError(f"Download failed {response.status_code}: {response.text[:1000]}")

    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


def list_sibling_items(access_token: str, item: Dict) -> List[Dict]:
    parent = item.get("parentReference", {})
    drive_id = parent.get("driveId")
    parent_id = parent.get("id")
    if not drive_id or not parent_id:
        return []

    data = graph_get(access_token, f"{GRAPH_API_ROOT}/drives/{drive_id}/items/{parent_id}/children")
    return data.get("value", [])


def find_related_transcript_item(access_token: str, recording_item: Dict) -> Optional[Dict]:
    siblings = list_sibling_items(access_token, recording_item)
    recording_stem = Path(recording_item.get("name", "")).stem.lower()
    transcript_exts = {".vtt", ".srt", ".txt"}

    transcript_candidates = []
    for item in siblings:
        name = item.get("name", "")
        suffix = Path(name).suffix.lower()
        if suffix not in transcript_exts:
            continue
        score = 1
        if recording_stem and recording_stem in Path(name).stem.lower():
            score += 2
        transcript_candidates.append((score, item))

    if not transcript_candidates:
        return None

    transcript_candidates.sort(key=lambda entry: entry[0], reverse=True)
    return transcript_candidates[0][1]


def download_teams_recording_assets(access_token: str, sharing_url: str, output_dir: Path) -> Dict[str, Optional[Path]]:
    recording_item = resolve_recording_link(access_token, sharing_url)
    video_path = download_drive_item(access_token, recording_item, output_dir)

    transcript_path = None
    transcript_item = find_related_transcript_item(access_token, recording_item)
    if transcript_item:
        transcript_path = download_drive_item(access_token, transcript_item, output_dir)

    return {
        "video_path": video_path,
        "transcript_path": transcript_path,
        "recording_name": recording_item.get("name", ""),
    }
