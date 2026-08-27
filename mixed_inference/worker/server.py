from __future__ import annotations

import json
import os
import re
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from ..core.errors import MIError
from .service import MIWorkerService


class MIWorkerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: MIWorkerService):
        super().__init__(address, MIWorkerRequestHandler)
        self.service = service


class MIWorkerRequestHandler(BaseHTTPRequestHandler):
    server: MIWorkerHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        try:
            path = self._path()
            if path == "/v1/health":
                self._json(HTTPStatus.OK, self.server.service.health())
                return
            if path == "/v1/capabilities":
                self._json(HTTPStatus.OK, self.server.service.capabilities())
                return
            if path == "/v1/sessions":
                self._json(
                    HTTPStatus.OK,
                    {"items": self.server.service.list_sessions()},
                )
                return
            match = re.fullmatch(r"/v1/sessions/([^/]+)", path)
            if match:
                self._json(
                    HTTPStatus.OK, self.server.service.session(match.group(1))
                )
                return
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/weights/([^/]+)/upload", path
            )
            if match:
                self._json(
                    HTTPStatus.OK,
                    self.server.service.upload_status(
                        match.group(1), match.group(2)
                    ),
                )
                return
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/weights/([^/]+)", path
            )
            if match:
                file_path, digest = self.server.service.weight_file(
                    match.group(1), match.group(2)
                )
                self._file(file_path, digest)
                return
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/rounds/(\d+)", path
            )
            if match:
                self._json(
                    HTTPStatus.OK,
                    self.server.service.round(
                        match.group(1), int(match.group(2))
                    ),
                )
                return
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/rounds/(\d+)/gradient", path
            )
            if match:
                file_path, digest = self.server.service.gradient_file(
                    match.group(1), int(match.group(2))
                )
                self._file(file_path, digest)
                return
            self._not_found(path)
        except BaseException as exc:
            self._error(exc)

    def do_POST(self) -> None:
        try:
            path = self._path()
            if path == "/v1/sessions":
                self._json(
                    HTTPStatus.CREATED,
                    self.server.service.create_session(self._read_json()),
                )
                return
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/rounds/(\d+)/compute", path
            )
            if match:
                self._json(
                    HTTPStatus.OK,
                    self.server.service.compute_round(
                        match.group(1), int(match.group(2)), self._read_json()
                    ),
                )
                return
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/rounds/(\d+)/ack", path
            )
            if match:
                self._json(
                    HTTPStatus.OK,
                    self.server.service.ack_round(
                        match.group(1), int(match.group(2)), self._read_json()
                    ),
                )
                return
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/actions/cancel", path
            )
            if match:
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.service.cancel(match.group(1)),
                )
                return
            self._not_found(path)
        except BaseException as exc:
            self._error(exc)

    def do_PUT(self) -> None:
        try:
            path = self._path()
            match = re.fullmatch(
                r"/v1/sessions/([^/]+)/weights/([^/]+)", path
            )
            if not match:
                self._not_found(path)
                return
            length = self._content_length()
            body = self.rfile.read(length)
            if len(body) != length:
                raise MIError(
                    "Weight request body ended early",
                    code="incomplete_request_body",
                    status=400,
                )
            offset, end, total = self._content_range(length)
            expected_sha = self.headers.get("X-Content-SHA256", "").strip().lower()
            if expected_sha.startswith("sha256:"):
                expected_sha = expected_sha[7:]
            final = end + 1 == total
            self._json(
                HTTPStatus.OK,
                self.server.service.upload_weights(
                    match.group(1),
                    match.group(2),
                    body,
                    offset=offset,
                    total_bytes=total,
                    expected_sha256=expected_sha,
                    final=final,
                    encoding=self.headers.get("Content-Encoding", "none").lower(),
                ),
            )
        except BaseException as exc:
            self._error(exc)

    def do_DELETE(self) -> None:
        try:
            path = self._path()
            match = re.fullmatch(r"/v1/sessions/([^/]+)", path)
            if not match:
                self._not_found(path)
                return
            self._json(
                HTTPStatus.OK,
                self.server.service.delete_session(match.group(1)),
            )
        except BaseException as exc:
            self._error(exc)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _path(self) -> str:
        return urlsplit(self.path).path.rstrip("/") or "/"

    def _read_json(self) -> dict[str, Any]:
        length = self._content_length()
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MIError(
                "Request body must be valid UTF-8 JSON",
                code="invalid_json",
                status=400,
            ) from exc
        if not isinstance(value, dict):
            raise MIError(
                "Request JSON root must be an object",
                code="invalid_json",
                status=400,
            )
        return value

    def _content_length(self) -> int:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise MIError(
                "Invalid Content-Length", code="invalid_content_length", status=400
            ) from exc
        if length < 0 or length > 8 * 1024**3:
            raise MIError(
                "Content-Length exceeds worker limit",
                code="request_too_large",
                status=413,
            )
        return length

    def _content_range(self, length: int) -> tuple[int, int, int]:
        value = self.headers.get("Content-Range")
        if not value:
            return 0, max(0, length - 1), length
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
        if not match:
            raise MIError(
                "Content-Range must use bytes start-end/total",
                code="invalid_content_range",
                status=400,
            )
        start, end, total = (int(item) for item in match.groups())
        if end < start or end - start + 1 != length or total < end + 1:
            raise MIError(
                "Content-Range does not match request body",
                code="invalid_content_range",
                status=400,
            )
        return start, end, total

    def _file(self, path: Path, digest: str) -> None:
        content_encoding = "identity"
        served_path = path
        accepted = {
            item.strip().lower()
            for item in self.headers.get("Accept-Encoding", "").split(",")
        }
        if "zstd" in accepted:
            served_path = self._zstd_file(path)
            content_encoding = "zstd"
        total = served_path.stat().st_size
        start, end = 0, max(0, total - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header.strip())
            if not match:
                raise MIError(
                    "Only a single bytes=start-end range is supported",
                    code="invalid_range",
                    status=416,
                )
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else total - 1
            if start >= total or end < start or end >= total:
                raise MIError(
                    "Requested byte range is outside the artifact",
                    code="range_not_satisfiable",
                    status=416,
                )
            status = HTTPStatus.PARTIAL_CONTENT
        length = 0 if total == 0 else end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", f'"sha256:{digest}"')
        self.send_header("X-Content-SHA256", digest)
        if content_encoding != "identity":
            self.send_header("Content-Encoding", content_encoding)
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        with served_path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    @staticmethod
    def _zstd_file(path: Path) -> Path:
        try:
            import zstandard
        except ImportError as exc:
            raise MIError(
                "zstandard is not installed on the worker",
                code="compression_unavailable",
                status=406,
            ) from exc
        encoded = path.with_suffix(path.suffix + ".zst")
        if encoded.is_file() and encoded.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            return encoded
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{encoded.name}.", suffix=".tmp", dir=encoded.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            with path.open("rb") as source, temporary_path.open("wb") as target:
                zstandard.ZstdCompressor(level=3).copy_stream(source, target)
            os.replace(temporary_path, encoded)
        finally:
            temporary_path.unlink(missing_ok=True)
        return encoded

    def _json(self, status: int | HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self, path: str) -> None:
        raise MIError(
            f"Worker route not found: {path}", code="route_not_found", status=404
        )

    def _error(self, error: BaseException) -> None:
        if isinstance(error, ValidationError):
            status = 422
            code = "validation_error"
            detail = "Worker request validation failed"
            fields: Any = error.errors()
        elif isinstance(error, MIError):
            status = error.status
            code = error.code
            detail = str(error)
            fields = None
        else:
            status = 500
            code = "internal_error"
            detail = f"{type(error).__name__}: {error}"
            fields = None
        try:
            self._json(
                status,
                {
                    "error": {
                        "code": code,
                        "detail": detail,
                        "status": status,
                        "fields": fields,
                    }
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve_worker(
    host: str,
    port: int,
    *,
    root: str | Path | None = None,
    require_rm: bool = True,
) -> None:
    service = MIWorkerService(root, require_rm=require_rm)
    server = MIWorkerHTTPServer((host, port), service)
    try:
        server.serve_forever()
    finally:
        server.server_close()
