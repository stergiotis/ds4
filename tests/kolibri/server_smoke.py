#!/usr/bin/env python3
"""Kolibri 1 through ds4-server's OpenAI-compatible endpoint.

    ./ds4-server --rocm -m gguf/Kolibri-1-F8.gguf --ctx 16384 &
    python3 tests/kolibri/server_smoke.py [--url http://127.0.0.1:8000]

1. German chat, reasoning off: German content, no reasoning.
2. English chat, reasoning low: reasoning and content arrive separately.
3. Tool round trip: the model calls get_weather with a city, gets the
   result back as a tool message and answers with the temperature.
4. The same English request streamed: reasoning deltas, then content.
"""
import argparse
import json
import sys
import time
import urllib.request

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}


def post(url, body, stream=False):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=900)
    if not stream:
        return json.load(resp), time.time() - t0
    events = []
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events, time.time() - t0


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    ok = True

    r, dt = post(args.url, {"model": "kolibri-1", "max_tokens": 200, "temperature": 0,
                            "reasoning_effort": "none",
                            "messages": [{"role": "user", "content": "Wie heißt die Hauptstadt von Bayern? Antworte in einem Satz."}]})
    msg = r["choices"][0]["message"]
    print(f"[de] {dt:.1f}s content={msg.get('content')!r} reasoning={msg.get('reasoning_content')!r}")
    ok &= check("München" in (msg.get("content") or ""), "German answer names München")
    ok &= check(not msg.get("reasoning_content"), "no reasoning with reasoning_effort=none")

    body = {"model": "kolibri-1", "max_tokens": 1500, "temperature": 0, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}]}
    r, dt = post(args.url, body)
    msg = r["choices"][0]["message"]
    print(f"[en] {dt:.1f}s finish={r['choices'][0]['finish_reason']} content={msg.get('content')!r}\n"
          f"     reasoning={(msg.get('reasoning_content') or '')[:160]!r}...")
    ok &= check("391" in (msg.get("content") or ""), "English answer is 391")
    ok &= check(bool(msg.get("reasoning_content")), "reasoning returned separately")
    ok &= check("<think>" not in (msg.get("content") or "") and "</think>" not in (msg.get("content") or ""),
                "no think tags in content")

    events, dt = post(args.url, dict(body, stream=True), stream=True)
    deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
    reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
    content = "".join(d.get("content") or "" for d in deltas)
    print(f"[stream] {dt:.1f}s {len(deltas)} deltas content={content!r}")
    ok &= check(bool(reasoning) and "391" in content, "streamed reasoning, then content")

    messages = [{"role": "user", "content": "What's the weather in Berlin right now? Use the tool."}]
    r, dt = post(args.url, {"model": "kolibri-1", "max_tokens": 1500, "temperature": 0,
                            "reasoning_effort": "low", "tools": [WEATHER], "messages": messages})
    choice = r["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    print(f"[tool] {dt:.1f}s finish={choice['finish_reason']} calls={json.dumps(calls)}")
    ok &= check(choice["finish_reason"] == "tool_calls" and len(calls) == 1, "one tool call")
    if calls:
        fn = calls[0]["function"]
        args_obj = json.loads(fn["arguments"])
        ok &= check(fn["name"] == "get_weather" and "berlin" in args_obj.get("city", "").lower(),
                    "get_weather(city=Berlin)")
        messages.append({"role": "assistant", "content": choice["message"].get("content") or "",
                         "reasoning_content": choice["message"].get("reasoning_content"),
                         "tool_calls": calls})
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                         "content": json.dumps({"temp_c": 18, "sky": "cloudy"})})
        r, dt = post(args.url, {"model": "kolibri-1", "max_tokens": 1500, "temperature": 0,
                                "reasoning_effort": "low", "tools": [WEATHER], "messages": messages})
        final = r["choices"][0]["message"].get("content") or ""
        print(f"[tool] {dt:.1f}s final={final!r}")
        ok &= check("18" in final, "final answer uses the tool result (18 °C)")
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
