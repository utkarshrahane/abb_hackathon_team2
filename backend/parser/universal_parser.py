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
        record: Dict[str, Any] = {}
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            return None

        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                record[k.strip().lower()] = v.strip()

        # try parse timestamp
        ts_val = record.get("timestamp") or record.get("time")
        ts = None
        if ts_val:
            ts = self._parse_iso_ts(ts_val)

        level = record.get("level", "UNKNOWN").upper()
        return {"timestamp": ts, "level": level, "message": block, **record}

    def parse_dotnet_line(self, line: str) -> Optional[Dict[str, Any]]:
        # dotnet lines are space-separated key=value tokens, timestamp is ts=...
        line = line.strip()
        if not line:
            return None

        # Special-case: if ' error=' exists, split it out to capture the rest (which may contain spaces)
        err_part = None
        if " error=" in line:
            line, err_part = line.split(" error=", 1)
            err_part = err_part.strip()

        record: Dict[str, Any] = {}
        # tokenise by spaces, parse key=value
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                record[k.strip().lower()] = v.strip()

        if err_part:
            record["error"] = err_part

        # timestamp
        ts = None
        m = self.iso_ts_re.search(line)
        if m:
            ts = self._parse_iso_ts(m.group(0))

        level = record.get("level", "UNKNOWN").upper()
        return {"timestamp": ts, "level": level, "message": line + (" error=" + err_part if err_part else ""), **record}

    def parse_angular_line(self, line: str) -> Optional[Dict[str, Any]]:
        m = self.apache_re.match(line.strip())
        if not m:
            return None
        groups = m.groupdict()
        ts = self._parse_apache_ts(groups.get("time"))
        level = groups.get("level", "UNKNOWN").upper()
        # include most useful fields
        return {
            "timestamp": ts,
            "level": level,
            "ip": groups.get("ip"),
            "method": groups.get("method"),
            "path": groups.get("path"),
            "status": groups.get("status"),
            "size": groups.get("size"),
            "pod": groups.get("pod"),
            "message": line.strip(),
        }

    # ---------- Top-level parsing API ----------
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

        # Fallback to very simple key:value or key=value on single line
        if ":" in line or "=" in line:
            rec = {}
            # simple kv pairs separated by whitespace or newlines
            for token in re.split(r"\s+", line):
                if ":" in token:
                    k, v = token.split(":", 1)
                    rec[k.strip().lower()] = v.strip()
                elif "=" in token:
                    k, v = token.split("=", 1)
                    rec[k.strip().lower()] = v.strip()

            ts = None
            m = self.iso_ts_re.search(line)
            if m:
                ts = self._parse_iso_ts(m.group(0))
            level = rec.get("level", "UNKNOWN").upper()
            return {"timestamp": ts, "level": level, "message": line.strip(), **rec}

        # Nothing recognizable — return raw message
        return {"timestamp": None, "level": "UNKNOWN", "message": line.strip()}

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
        return pd.DataFrame(all_records)
