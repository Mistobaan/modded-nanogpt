from pathlib import Path
import zipfile
import requests


def zip_log_directory(log_dir: Path, archive_path: Path) -> Path:
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log path {log_dir} is not a directory")

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in log_dir.rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(log_dir))
    return archive_path


def upload_archive(archive_path: Path, upload_url: str, api_key: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    with archive_path.open("rb") as handle:
        files = {"file": (archive_path.name, handle, "application/zip")}
        response = requests.post(upload_url, headers=headers, files=files, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Upload failed ({response.status_code}): {response.text.strip()}"
        )
