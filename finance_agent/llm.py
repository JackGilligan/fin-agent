from __future__ import annotations

import json
import re
import requests
import os
from pathlib import Path
from typing import Any, Iterable


class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaLLMClient(LLMClient):
    def __init__(self, model: str = "llama3.1:8b", base_url: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response") or "")


class ScriptedLLMClient(LLMClient):
    def __init__(self, responses: Iterable[str | dict[str, Any]] | None = None):
        self._responses = list(responses or [])

    def complete(self, prompt: str) -> str:
        if not self._responses:
            return json.dumps({"tool": "compute_totals", "arguments": {}})
        next_response = self._responses.pop(0)
        if isinstance(next_response, str):
            return next_response
        return json.dumps(next_response)


class ReplayLLMClient(LLMClient):
    def __init__(self, recording_path: str | Path):
        self.recording_path = Path(recording_path)
        self._responses = self._load_recording()
        self._index = 0

    def _load_recording(self) -> list[Any]:
        if not self.recording_path.exists():
            return []
        with self.recording_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "responses" in payload:
            return payload["responses"]
        if isinstance(payload, list):
            return payload
        return []

    def complete(self, prompt: str) -> str:
        if self._index >= len(self._responses):
            return json.dumps({"tool": "write_report", "arguments": {}})
        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, dict):
            if "response" in response:
                return str(response["response"])
            if "tool" in response:
                return json.dumps({"tool": response["tool"], "arguments": response.get("arguments", {})})
        return str(response)


class RecordingLLMClient(LLMClient):
    def __init__(self, base_client: LLMClient, recording_path: str | Path):
        self.base_client = base_client
        self.recording_path = Path(recording_path)
        self._history: list[dict[str, str]] = []

    def complete(self, prompt: str) -> str:
        response = self.base_client.complete(prompt)
        self._history.append({"prompt": prompt, "response": response})
        self._persist()
        return response

    def _persist(self) -> None:
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"responses": self._history}
        with self.recording_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


def parse_json_response(raw_response: str) -> Any | None:
    text = (raw_response or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            return parsed
        except json.JSONDecodeError:
            continue
    return None


def parse_tool_choice(raw_response: str) -> dict[str, Any]:
    text = (raw_response or "").strip()
    if not text:
        return {"tool": "invalid", "arguments": {}}
    parsed = parse_json_response(text)
    if isinstance(parsed, dict):
        return {"tool": str(parsed.get("tool") or parsed.get("name") or "invalid"), "arguments": parsed.get("arguments") or parsed.get("params") or {}}
    if isinstance(parsed, str):
        return {"tool": parsed, "arguments": {}}
    match = re.search(r"(?:tool|name)\s*[:=]\s*['\"]?([A-Za-z_\-]+)", text, re.IGNORECASE)
    if match:
        return {"tool": match.group(1), "arguments": {}}
    return {"tool": "invalid", "arguments": {}}
