# backend/parser/universal_parser.py

import os
import re
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List


class UniversalLogParser:
    """A parser that heuristically parses kube (multi-line), dotnet (key=value),
    and angular (apache-style) logs.

    It keeps a best-effort approach: returns dicts with parsed fields when possible
    and always includes the raw message under 'message'.
    """

    def __init__(self):
        # ISO-like timestamps (used by dotnet samples)
        self.iso_ts_re = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:[+-]\d{2}:\d{2}|Z)?")
        # split separator for kube multiline blocks
        self.kube_sep_re = re.compile(r"^-{4,}$", re.MULTILINE)
        # apache/nginx combined-ish line
        self.apache_re = re.compile(
            r'(?P<ip>\S+)\s+-\s+-\s+\[(?P<time>[^\]]+)\]\s+"(?P<method>\S+)\s+(?P<path>\S+)\s+[^\"]+"\s+(?P<status>\d{3})\s+(?P<size>\d+)\s+"(?P<pod>[^"]+)"\s+"(?P<level>[^"]+)"'
        )

    def _normalize_level(self, level: Optional[str]) -> str:
        """Normalize log level to standard values."""
        if not level:
            return "UNKNOWN"
        level = level.upper()
        if level in {"INFO", "DEBUG", "WARN", "WARNING", "ERROR", "TRACE", "FATAL"}:
            return "WARNING" if level == "WARN" else level
        return "UNKNOWN"

    def _make_base_record(self, message: str, source: str = "other") -> Dict[str, Any]:
        """Create a base record with standard fields."""
        return {
            "timestamp": None,
            "level": "UNKNOWN",
            "source": source,
            "pod": None,
            "message": message,
            "status": None,
            "error": None,
            "event": None,
            "metadata": {},
        }

    # ---------- Helpers for timestamp parsing ----------
    def _parse_iso_ts(self, ts_raw: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            return None

    def _parse_apache_ts(self, ts_raw: str) -> Optional[datetime]:
        # Example: 12/Sep/2025:23:59:33 +0530
        try:
            return datetime.strptime(ts_raw, "%d/%b/%Y:%H:%M:%S %z")
        except Exception:
            return None

    # ---------- Format-specific parsers ----------
    def parse_kube_block(self, block: str) -> Optional[Dict[str, Any]]:
        # Kube block is multiple lines like "KEY: VALUE"
        record = self._make_base_record(block, source="kube")
        kv: Dict[str, str] = {}
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            return None

        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip().lower()] = v.strip()

        # try parse timestamp
        ts_val = kv.get("timestamp") or kv.get("time")
        if ts_val:
            record["timestamp"] = self._parse_iso_ts(ts_val)

        record["level"] = self._normalize_level(kv.get("level"))
        record["pod"] = kv.get("pod")
        record["event"] = kv.get("event")
        record["status"] = kv.get("reason")  # kube uses "reason" as status
        record["metadata"] = {k: v for k, v in kv.items() if k not in {
            "timestamp", "time", "level", "pod", "event", "reason"
        }}
        return record

    def parse_dotnet_line(self, line: str) -> Optional[Dict[str, Any]]:
        # dotnet lines are space-separated key=value tokens, timestamp is ts=...
        line = line.strip()
        if not line:
            return None

        record = self._make_base_record(line, source="dotnet")

        # Special-case: if ' error=' exists, split it out to capture the rest (which may contain spaces)
        err_part = None
        if " error=" in line:
            line, err_part = line.split(" error=", 1)
            err_part = err_part.strip()

        kv: Dict[str, str] = {}
        # tokenise by spaces, parse key=value
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                kv[k.strip().lower()] = v.strip()

        if err_part:
            record["error"] = err_part

        # timestamp
        m = self.iso_ts_re.search(line)
        if m:
            record["timestamp"] = self._parse_iso_ts(m.group(0))

        record["level"] = self._normalize_level(kv.get("level"))
        record["pod"] = kv.get("pod")
        record["event"] = kv.get("event")
        record["status"] = kv.get("status")
        record["metadata"] = {k: v for k, v in kv.items() if k not in {
            "ts", "level", "pod", "event", "status", "error"
        }}
        return record

    def parse_angular_line(self, line: str) -> Optional[Dict[str, Any]]:
        m = self.apache_re.match(line.strip())
        if not m:
            return None

        record = self._make_base_record(line, source="angular")
        groups = m.groupdict()

        record.update({
            "timestamp": self._parse_apache_ts(groups.get("time")),
            "level": self._normalize_level(groups.get("level")),
            "pod": groups.get("pod"),
            "status": groups.get("status"),
            # put apache-specific fields in metadata
            "metadata": {
                "ip": groups.get("ip"),
                "method": groups.get("method"),
                "path": groups.get("path"),
                "size": groups.get("size"),
            }
        })
        return record

    # ---------- Top-level parsing API ----------
    def parse_unknown_format(self, line: str) -> Dict[str, Any]:
        """Fallback parser for unknown log formats. Attempts to extract common fields
        like timestamp and level, but always returns a valid record.
        """
        record = self._make_base_record(line, source="unknown")

        # try to find an ISO timestamp anywhere in the line
        m = self.iso_ts_re.search(line)
        if m:
            record["timestamp"] = self._parse_iso_ts(m.group(0))

        # look for common level indicators
        for level in ["INFO", "DEBUG", "WARN", "WARNING", "ERROR", "TRACE", "FATAL"]:
            if f"[{level}]" in line or f" {level} " in line or f"level={level}" in line:
                record["level"] = self._normalize_level(level)
                break

        # extract any key=value or key:value pairs into metadata
        metadata = {}
        for pair_re in [
            # key=value
            r'([a-zA-Z][a-zA-Z0-9_-]+)=([^"\s][^\s]*)',
            # key="value with spaces"
            r'([a-zA-Z][a-zA-Z0-9_-]+)="([^"]+)"',
            # key:value
            r'([a-zA-Z][a-zA-Z0-9_-]+):\s*([^,}\s][^,}]*)',
        ]:
            for k, v in re.findall(pair_re, line):
                k = k.lower()
                if k == "level":
                    record["level"] = self._normalize_level(v)
                elif k in {"pod", "container", "service"}:
                    record["pod"] = v
                elif k in {"status", "code", "result"}:
                    record["status"] = v
                elif k == "error":
                    record["error"] = v
                elif k == "event":
                    record["event"] = v
                else:
                    metadata[k] = v

        if metadata:
            record["metadata"] = metadata

        return record

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        # Try angular first (distinct format)
        a = self.parse_angular_line(line)
        if a:
            return a

        # Then dotnet-style (ts=...) single-line
        if "ts=" in line or self.iso_ts_re.search(line):
            d = self.parse_dotnet_line(line)
            if d:
                return d

        # Unknown format - use fallback parser
        return self.parse_unknown_format(line)

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read a file and detect whether it contains kube-style multi-line entries or
        single-line logs. Returns a list of record dicts.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        records: List[Dict[str, Any]] = []

        # Heuristic: if file contains lines starting with 'TIMESTAMP:' treat as kube blocks
        if "TIMESTAMP:" in content or "POD:" in content:
            # split by separator lines of dashes (4 or more)
            blocks = re.split(r"\n-{4,}\n", content)
            for block in blocks:
                parsed = self.parse_kube_block(block)
                if parsed:
                    parsed["source_file"] = os.path.basename(file_path)
                    records.append(parsed)
            return records

        # Otherwise interpret as single-line logs (one per line)
        for line in content.splitlines():
            parsed = self.parse_line(line)
            if parsed:
                parsed["source_file"] = os.path.basename(file_path)
                records.append(parsed)

        return records

    def parse_folder(self, folder_path: str) -> pd.DataFrame:
        all_records: List[Dict[str, Any]] = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".log"):
                file_path = os.path.join(folder_path, file_name)
                all_records.extend(self.parse_file(file_path))
        df = pd.DataFrame(all_records)

        # Normalize timestamps to a single canonical form for querying:
        # - `timestamp`: pandas datetime64[ns, UTC] (or NaT)
        # - `timestamp_iso`: RFC3339/ISO8601 string in UTC ending with Z (or None)
        # - `timestamp_epoch_ms`: integer milliseconds since epoch (or None)
        if 'timestamp' in df.columns:
            # convert to pandas datetime (coerce invalid -> NaT) and normalize to UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

            def _to_iso(x):
                if pd.isna(x):
                    return None
                try:
                    s = x.isoformat()
                except Exception:
                    s = str(x)
                # make UTC look like ...Z instead of +00:00
                if s.endswith('+00:00'):
                    s = s.replace('+00:00', 'Z')
                return s

            df['timestamp_iso'] = df['timestamp'].apply(_to_iso)

            def _to_epoch_ms(x):
                if pd.isna(x):
                    return None
                try:
                    return int(x.timestamp() * 1000)
                except Exception:
                    return None

            df['timestamp_epoch_ms'] = df['timestamp'].apply(_to_epoch_ms)

        return df
