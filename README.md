# ollama-rename

Interactive, safe “rename” for Ollama models. Under the hood it **copies** a model to a new name, then (optionally) **deletes** the original.

## Features
- Lists local models, fuzzy-pick one, suggests a short clean name.
- Copy → optional delete (i.e., move). Checks if model is loaded before delete.
- Overwrite guard (interactive prompt or `--overwrite` flag).
- Works via Ollama HTTP API; can fall back to `ollama cp`/`ollama rm`.
- Windows & Linux.

## Requirements
- Ollama installed and running (`OLLAMA_HOST` honored, defaults to `http://127.0.0.1:11434`).
- Rust toolchain (for building from source).

## Install (Windows)
Double-click `install.bat` or run:
```powershell
.\install.bat
```

It builds the binary and places it in `%USERPROFILE%\bin` (adds to PATH if needed).

## Install (Linux, from source)

```bash
cargo build --release
install -Dm755 target/release/ollama-rename ~/.local/bin/ollama-rename
```

## Usage

Interactive (recommended):

```bash
ollama-rename
```

Non-interactive:

```bash
# copy
ollama-rename rename --from "hf.co/NikolayKozloff/NextCoder-14B-Q4_K_M-GGUF:Q4_K_M" --to "NextCoder"

# move (copy + delete original)
ollama-rename rename --from "qwen3-coder:latest" --to "qwen3-coder" --delete-original

# replace destination if it exists
ollama-rename rename --from "gpt-oss:latest" --to "gpt-oss" --overwrite
```

Useful flags: `--host <URL>`, `--use-cli-fallback`, `--force` (delete even if loaded), `--dry-run`.
