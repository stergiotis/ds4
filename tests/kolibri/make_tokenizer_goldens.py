#!/usr/bin/env python3
"""Golden token ids for Kolibri 1, from HF `tokenizers` and the released chat
template. The C tokenizer and chat renderer must reproduce these exactly.

    uv run make_tokenizer_goldens.py --model-dir DIR [--out ../golden/kolibri/tokenizer.json]

DIR holds tokenizer.json, tokenizer_config.json (the HF snapshot directory).
"""

import argparse
import json
import os

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

# Raw strings: they exercise the pre-tokenizer regex (single-digit split,
# contractions, non-letter prefix before letters, whitespace runs, newlines),
# byte-level fallback for unusual bytes, and special-token matching.
RAW = [
    ("empty", ""),
    ("hello", "Hello world"),
    ("german", "Grüß Gott! Wie geht's dir heute? Die Straße ist naß."),
    ("digits", "In 2026, 1234567 + 89 = 1234656; pi≈3.14159"),
    ("contractions", "I'm sure they'll say it's what we'd've DONE, isn't it?"),
    ("whitespace", "a  b   c\t\td\n\ne \n  f\r\ng    "),
    ("code", "def f(x):\n    return x**2  # square\n\nprint(f(3))\n"),
    ("unicode", "日本語のテキスト, emoji 🦜🐦, Ελληνικά, العربية"),
    ("specials", "<|im_start|>user\nHi<|im_end|>\n<think>\n\n</think>\n\n"),
    ("tool_tags", "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"),
    ("punct_prefix", "(Hello) [world] {foo} \"bar\" 'baz' -qux _x"),
    ("long_word", "Donaudampfschifffahrtsgesellschaftskapitänsmütze"),
    # Combining marks (\p{M}): Kolibri's regex does not join them to letters.
    ("marks", "Cafe\u0301 nai\u0308ve \u0301x Zu\u0308rich"),
    ("devanagari", "नमस्ते दुनिया, यह एक परीक्षण है।"),
    ("thai_hebrew", "สวัสดีครับ שָׁלוֹם"),
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

CHAT = [
    ("chat_en_default",
     [{"role": "user", "content": "What is the capital of France?"}], None, {}),
    ("chat_de_nothink",
     [{"role": "user", "content": "Erkläre kurz, was ein Kolibri ist."}],
     None, {"enable_thinking": False}),
    ("chat_system_low",
     [{"role": "system", "content": "You are a terse assistant."},
      {"role": "user", "content": "Name three primes."}],
     None, {"reasoning_effort": "low"}),
    ("chat_medium",
     [{"role": "user", "content": "Why is the sky blue?"}],
     None, {"reasoning_effort": "medium"}),
    ("chat_none_effort",
     [{"role": "user", "content": "Say hi."}], None, {"reasoning_effort": "none"}),
    ("chat_multiturn",
     [{"role": "user", "content": "Hi!"},
      {"role": "assistant", "content": "Hello! How can I help?",
       "reasoning": "The user greets me."},
      {"role": "user", "content": "Tell me a joke."}],
     None, {}),
    ("chat_tool_call_roundtrip",
     [{"role": "user", "content": "What's the weather in Berlin?"},
      {"role": "assistant", "content": "", "reasoning": "I should call the tool.",
       "tool_calls": [{"type": "function", "function": {
           "name": "get_weather", "arguments": {"city": "Berlin"}}}]},
      {"role": "tool", "content": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"}],
     [WEATHER_TOOL], {}),
]


def render(template: str, messages, tools, kwargs) -> str:
    # Matches transformers' apply_chat_template environment.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=messages, tools=tools, add_generation_prompt=True, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "golden", "kolibri", "tokenizer.json"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    cfg = json.load(open(os.path.join(args.model_dir, "tokenizer_config.json")))
    template = cfg["chat_template"]

    out = {"raw": [], "chat": []}
    for name, text in RAW:
        ids = tok.encode(text, add_special_tokens=False).ids
        assert tok.decode(ids, skip_special_tokens=False) == text, name
        out["raw"].append({"name": name, "text": text, "ids": ids})
    for name, messages, tools, kwargs in CHAT:
        text = render(template, messages, tools, kwargs)
        ids = tok.encode(text, add_special_tokens=False).ids
        out["chat"].append({"name": name, "messages": messages, "tools": tools,
                            "kwargs": kwargs, "rendered": text, "ids": ids})
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {len(out['raw'])} raw + {len(out['chat'])} chat cases to {args.out}")


if __name__ == "__main__":
    main()
