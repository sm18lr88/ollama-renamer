use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, FuzzySelect, Input};
use regex::Regex;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use std::io::{self, Read};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "ollama-rename",
    version,
    about = "Interactive, safe model renamer for Ollama"
)]
struct Cli {
    /// Set Ollama base URL (e.g. http://127.0.0.1:11434). Falls back to OLLAMA_HOST or http://127.0.0.1:11434.
    #[arg(long)]
    host: Option<String>,

    /// Use the Ollama CLI as a fallback if API calls fail (runs `ollama cp`/`ollama rm`)
    #[arg(long, action=ArgAction::SetTrue)]
    use_cli_fallback: bool,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Non-interactive rename (copy + optional delete)
    Rename {
        /// Source model (as shown by `ollama list`, e.g. "hf.co/...:Q4_K_M" or "qwen3-coder:latest")
        #[arg(long)]
        from: String,
        /// Destination model name (e.g. "NextCoder" or "myspace/nextcoder:latest")
        #[arg(long)]
        to: String,
        /// Delete the original after copy succeeds (acts like move)
        #[arg(long, action=ArgAction::SetTrue)]
        delete_original: bool,
        /// Force delete even if model appears loaded (not recommended)
        #[arg(long, action=ArgAction::SetTrue)]
        force: bool,
        /// Dry-run: show what would happen
        #[arg(long, action=ArgAction::SetTrue)]
        dry_run: bool,
        /// Overwrite destination if it already exists (delete then copy)
        #[arg(long, action=ArgAction::SetTrue)]
        overwrite: bool,
    },
}

#[derive(Deserialize, Debug)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize, Debug, Clone)]
struct ModelInfo {
    name: String,
    #[serde(default)]
    size: Option<Value>,
    #[serde(default)]
    modified_at: Option<String>,
}

#[derive(Deserialize, Debug)]
struct PsResponse {
    models: Option<Vec<RunningModel>>,
}

#[derive(Deserialize, Debug)]
struct RunningModel {
    name: Option<String>,
}

fn main() {
    if let Err(e) = run_app() {
        eprintln!("\n{}", style(format!("Error: {:?}", e)).red().bold());
        pause_at_end();
    }
}

fn run_app() -> Result<()> {
    let cli = Cli::parse();
    let base = pick_base_url(cli.host.as_deref());
    let client = Client::builder()
        .timeout(Duration::from_secs(10)) // fast fail for normal calls
        .build()?;

    ensure_ollama_is_running(&client, &base)?;

    if let Some(Cmd::Rename {
        from,
        to,
        delete_original,
        force,
        dry_run,
        overwrite,
    }) = cli.command
    {
        run_non_interactive(
            &client,
            &base,
            &from,
            &to,
            delete_original,
            force,
            dry_run,
            cli.use_cli_fallback,
            overwrite,
        )
    } else {
        run_interactive(&client, &base, cli.use_cli_fallback)
    }
}

fn pause_at_end() {
    // Pause only if the app was likely double-clicked (no parent process or parent is explorer.exe)
    // This is a heuristic. A more robust method might involve checking for an allocated console.
    let should_pause = env::var("TERM").is_err() && env::var("PROMPT").is_err();

    if should_pause {
        println!("\nPress Enter to exit...");
        let _ = io::stdin().read(&mut [0u8]);
    }
}

fn pick_base_url(arg_host: Option<&str>) -> String {
    // Priority: --host > OLLAMA_HOST > default
    let host = arg_host
        .map(|s| s.to_string())
        .or_else(|| env::var("OLLAMA_HOST").ok())
        .unwrap_or_else(|| "127.0.0.1:11434".to_string());

    if host.starts_with("http://") || host.starts_with("https://") {
        host
    } else {
        format!("http://{}", host)
    }
}

fn run_interactive(client: &Client, base: &str, use_cli_fallback: bool) -> Result<()> {
    let theme = ColorfulTheme::default();
    println!(
        "{}",
        style("Ollama model renamer (safe copy → optional delete)").bold()
    );

    let mut models =
        list_models(client, base).context("Failed to list models. Is Ollama running?")?;
    if models.is_empty() {
        bail!("No models found. Use `ollama pull ...` first.");
    }
    // Newest modified first, then name
    models.sort_by(|a, b| {
        b.modified_at
            .cmp(&a.modified_at)
            .then_with(|| a.name.cmp(&b.name))
    });

    // Show list with fuzzy select
    let items: Vec<String> = models.iter().map(|m| format_model(m)).collect();
    let idx = FuzzySelect::with_theme(&theme)
        .with_prompt("Select a model to rename (copy)")
        .items(&items)
        .default(0)
        .interact()?;

    let chosen = &models[idx];
    println!("Selected: {}", style(&chosen.name).green());

    let suggested = suggest_simple_name(&chosen.name);
    let new_name: String = Input::with_theme(&theme)
        .with_prompt("New model name")
        .with_initial_text(&suggested)
        .validate_with(|input: &String| validate_model_name(input))
        .interact_text()?;
    let new_name = new_name.trim().to_string();

    if new_name == chosen.name {
        bail!("Destination name equals source; nothing to do.");
    }

    // Prevent accidental overwrite
    if model_exists(client, base, &new_name)? {
        let overwrite = Confirm::with_theme(&theme)
            .with_prompt(format!(
                "'{}' already exists. Overwrite (delete it first)?",
                &new_name
            ))
            .default(false)
            .interact()?;
        if !overwrite {
            println!("{}", style("Aborted (destination exists).").yellow());
            return Ok(());
        }
        delete_model(client, base, &new_name, use_cli_fallback)
            .with_context(|| format!("Failed to delete existing destination '{}'", &new_name))?;
    }

    println!(
        "{} {} -> {}",
        style("Copying").cyan().bold(),
        style(&chosen.name).yellow(),
        style(&new_name).yellow()
    );
    copy_model(client, base, &chosen.name, &new_name, use_cli_fallback)
        .with_context(|| format!("Copy failed from '{}' to '{}'", &chosen.name, &new_name))?;
    println!("{}", style("Copy OK.").green());

    // Offer delete
    let delete = Confirm::with_theme(&theme)
        .with_prompt(format!("Delete original '{}' (i.e., move)?", &chosen.name))
        .default(false)
        .interact()?;

    if delete {
        if model_is_running(client, base, &chosen.name).unwrap_or(false) {
            let proceed = Confirm::with_theme(&theme)
                .with_prompt(
                    "Model seems loaded (`ollama ps`). Stop it first. Proceed with delete anyway?",
                )
                .default(false)
                .interact()?;
            if !proceed {
                println!("{}", style("Skipped delete.").yellow());
                return Ok(());
            }
        }
        delete_model(client, base, &chosen.name, use_cli_fallback)
            .with_context(|| format!("Failed to delete '{}'", &chosen.name))?;
        println!("{}", style("Deleted original.").green());
    } else {
        println!("{}", style("Kept original (alias copy).").yellow());
    }

    println!(
        "\n{}  You can now use: {}",
        style("Done.").bold(),
        style(&new_name).bold().green()
    );
    Ok(())
}

fn run_non_interactive(
    client: &Client,
    base: &str,
    from: &str,
    to: &str,
    delete_original: bool,
    force: bool,
    dry_run: bool,
    use_cli_fallback: bool,
    overwrite: bool,
) -> Result<()> {
    // Normalize/trim
    let to = to.trim();
    let from = from.trim();
    validate_model_name(to).map_err(|e| anyhow::anyhow!(e))?;

    if dry_run {
        println!("[dry-run] Would copy '{}' -> '{}'", from, to);
        if delete_original {
            println!("[dry-run] Would delete original '{}'", from);
        }
        return Ok(());
    }

    if model_exists(client, base, to)? {
        if overwrite {
            delete_model(client, base, to, use_cli_fallback)
                .with_context(|| format!("Failed to delete existing destination '{}'", to))?;
        } else {
            bail!(
                "Destination '{}' already exists. Use --overwrite to replace it.",
                to
            );
        }
    }

    copy_model(client, base, from, to, use_cli_fallback)
        .with_context(|| format!("Copy failed from '{}' to '{}'", from, to))?;
    println!("{}", style("Copy OK.").green());

    if delete_original {
        if !force && model_is_running(client, base, from).unwrap_or(false) {
            bail!("Model appears loaded (via /api/ps). Use --force to attempt delete anyway, or stop it first.");
        }
        delete_model(client, base, from, use_cli_fallback)
            .with_context(|| format!("Failed to delete '{}'", from))?;
        println!("{}", style("Deleted original.").green());
    }

    Ok(())
}

fn validate_model_name(s: &str) -> std::result::Result<(), String> {
    // Require non-empty path segments; optional :tag
    let re =
        Regex::new(r#"^(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:[:][A-Za-z0-9._-]+)?$"#).unwrap();
    if !re.is_match(s) {
        return Err("Invalid name. Use letters, numbers, . _ - / and optional :tag".into());
    }
    Ok(())
}

fn suggest_simple_name(full: &str) -> String {
    // 1) strip tag
    let before_tag = full.split(':').next().unwrap_or(full);
    // 2) take last path segment
    let last = before_tag.split('/').last().unwrap_or(before_tag);
    // 3) drop common suffix noise (very conservative)
    let mut s = last.to_string();
    for pat in &[
        "-GGUF", "-gguf", ".gguf", "-Q2", "-Q3", "-Q4", "-Q5", "-Q6", "-Q8", "-K", "_K", "-KM",
        "_KM", "-K_M", "_K_M", "-Q4_K", "_Q4_K", "-Q5_K", "_Q5_K",
    ] {
        if let Some(pos) = s.find(pat) {
            s.truncate(pos);
        }
    }
    s
}

fn api_url(base: &str, tail: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        tail.trim_start_matches('/')
    )
}

fn list_models(client: &Client, base: &str) -> Result<Vec<ModelInfo>> {
    // GET /api/tags returns { "models": [ { "name": "...", ... } ] }
    let url = api_url(base, "/api/tags");
    let resp = client.get(&url).send().context("GET /api/tags failed")?;
    if !resp.status().is_success() {
        bail!("GET /api/tags -> HTTP {}", resp.status());
    }
    let tr: TagsResponse = resp.json().context("Decode /api/tags JSON")?;
    Ok(tr.models)
}

fn ensure_ollama_is_running(client: &Client, base: &str) -> Result<()> {
    if is_ollama_api_running(client, base) {
        return Ok(());
    }

    println!(
        "{}",
        style("Ollama API not responsive. Checking CLI...").yellow()
    );

    match Command::new("ollama").arg("--version").output() {
        Ok(_) => {
            println!(
                "{}",
                style("Ollama CLI found. Attempting to start the service...").green()
            );
            start_ollama_service()?;

            println!("Waiting for Ollama to start...");
            for _ in 0..30 {
                if is_ollama_api_running(client, base) {
                    println!("{}", style("Ollama started successfully.").green());
                    return Ok(());
                }
                thread::sleep(Duration::from_secs(1));
            }
            bail!("Failed to start Ollama service (timeout).");
        }
        Err(_) => {
            bail!("Ollama CLI not found. Please install Ollama and ensure it's in your PATH.");
        }
    }
}

fn is_ollama_api_running(client: &Client, base: &str) -> bool {
    let url = api_url(base, "/api/version");
    client
        .get(&url)
        .timeout(Duration::from_secs(3))
        .send()
        .is_ok()
}

fn start_ollama_service() -> Result<()> {
    if cfg!(target_os = "windows") {
        // Try the Windows service first (name 'Ollama' from the official installer)
        if Command::new("sc")
            .args(["start", "Ollama"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Ok(());
        }
        // Fallback: spawn a new window running `ollama serve`
        Command::new("cmd")
            .args(&["/C", "start", "ollama", "serve"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to start Ollama on Windows.")?;
    } else {
        Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to start Ollama.")?;
    }
    Ok(())
}

fn model_is_running(client: &Client, base: &str, name: &str) -> Result<bool> {
    // GET /api/ps
    let url = api_url(base, "/api/ps");
    let resp = client.get(&url).send().context("GET /api/ps failed")?;
    if !resp.status().is_success() {
        return Ok(false); // if missing, don't block
    }
    let pr: PsResponse = resp.json().context("Decode /api/ps JSON")?;
    let loaded = pr
        .models
        .unwrap_or_default()
        .into_iter()
        .filter_map(|m| m.name)
        .any(|n| n == name);
    Ok(loaded)
}

fn copy_model(
    client: &Client,
    base: &str,
    from: &str,
    to: &str,
    use_cli_fallback: bool,
) -> Result<()> {
    // Prefer API: POST /api/copy {source,destination}
    let url = api_url(base, "/api/copy");
    let res = client
        .post(&url)
        .json(&json!({"source": from, "destination": to}))
        .timeout(Duration::from_secs(60 * 60)) // large copies can take a while
        .send();

    match res {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            if use_cli_fallback {
                eprintln!("API copy failed ({}). Falling back to CLI...", status);
                cli_copy(from, to)?;
                Ok(())
            } else {
                bail!("API copy failed: status {} body {}", status, body);
            }
        }
        Err(e) => {
            if use_cli_fallback {
                eprintln!("API copy error: {}. Falling back to CLI...", e);
                cli_copy(from, to)?;
                Ok(())
            } else {
                Err(e).context("POST /api/copy failed")
            }
        }
    }
}

fn delete_model(client: &Client, base: &str, name: &str, use_cli_fallback: bool) -> Result<()> {
    // Some versions use DELETE /api/delete, some switched to POST. Try both.
    let url = api_url(base, "/api/delete");
    let try_delete = || -> Result<()> {
        let resp = client
            .delete(&url)
            .json(&json!({"model": name}))
            .timeout(Duration::from_secs(60))
            .send()?;
        if resp.status().is_success() {
            return Ok(());
        }
        // Try POST fallback:
        let resp2 = client
            .post(&url)
            .json(&json!({"model": name}))
            .timeout(Duration::from_secs(60))
            .send()?;
        if resp2.status().is_success() {
            Ok(())
        } else {
            bail!("Delete failed: {} / {}", resp.status(), resp2.status());
        }
    };

    match try_delete() {
        Ok(()) => Ok(()),
        Err(e) => {
            if use_cli_fallback {
                eprintln!("API delete failed ({}). Falling back to CLI...", e);
                cli_rm(name)?;
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

fn cli_copy(from: &str, to: &str) -> Result<()> {
    let status = Command::new("ollama")
        .args(["cp", from, to])
        .status()
        .context("Failed to invoke `ollama` binary")?;
    if !status.success() {
        bail!("`ollama cp` returned non-zero status");
    }
    Ok(())
}

fn cli_rm(name: &str) -> Result<()> {
    let status = Command::new("ollama")
        .args(["rm", name])
        .status()
        .context("Failed to invoke `ollama` binary")?;
    if !status.success() {
        bail!("`ollama rm` returned non-zero status");
    }
    Ok(())
}

fn format_model(m: &ModelInfo) -> String {
    let mut s = m.name.clone();
    if let Some(szv) = &m.size {
        if let Some(sz_str) = fmt_size_value(szv) {
            s.push_str(&format!("  ({})", sz_str));
        }
    }
    if let Some(modified) = &m.modified_at {
        s.push_str(&format!("  • {}", modified));
    }
    s
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn fmt_size_value(v: &Value) -> Option<String> {
    match v {
        Value::Number(n) => n.as_u64().map(format_size),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn model_exists(client: &Client, base: &str, name: &str) -> Result<bool> {
    let list = list_models(client, base)?;
    Ok(list.iter().any(|m| m.name == name))
}
