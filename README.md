# ollama-rename

Interactive, safe "rename" for Ollama models. Under the hood it **copies** a model to a new name, then (optionally) **deletes** the original.

<img width="598" height="154" alt="image" src="https://github.com/user-attachments/assets/0d7b870c-aadf-4c7d-80f9-5c2eceb51582" />


## Features
- Alpha software: use at your own risk.
- Fuzzy-pick a model, get a clean suggested name, and copy it safely.
- Copy + optional delete (i.e., move) with running-model guard and `--yes` auto-confirm.
- New subcommands: `list` (pretty or `--output json`) and `doctor` (health/version check).
- JSON output for scripting via `--output json`; CLI fallback via `--use-cli-fallback`.
- Uses Ollama APIs (`/api/copy`, `/api/delete`, `/api/tags`, `/api/ps`, `/api/version`) with CLI fallbacks.
- Windows & Linux.

## Requirements
- Ollama installed.
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
# Launch the guided picker (safe by default)
ollama-rename
# Tip: Press Esc to cancel at any prompt.
```

Non-interactive rename (copy/move):

```bash
# copy
ollama-rename rename --from "hf.co/NikolayKozloff/NextCoder-14B-Q4_K_M-GGUF:Q4_K_M" --to "NextCoder"

# move (copy + delete original)
ollama-rename rename --from "qwen3-coder:latest" --to "qwen3-coder" --delete-original

# replace destination if it exists
ollama-rename rename --from "gpt-oss:latest" --to "gpt-oss" --overwrite

# JSON report for scripts
ollama-rename rename --from "gpt-oss:latest" --to "gpt-oss" --output json
```

Other commands:

```bash
# list models (marking running ones)
ollama-rename list
ollama-rename list --running-only --output json

# quick health/version check
ollama-rename doctor
```

Useful flags: `--host <URL>`, `--use-cli-fallback`, `--force` (delete even if loaded), `--dry-run`, `--yes` (auto-confirm), `--output json`.

## Development

- Format: `cargo fmt`
- Test (with mocks): `cargo test`
- Release build: `cargo build --release`



