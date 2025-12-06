use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, FuzzySelect, Input};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::env;
use std::io::{self, Read};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum, Default)]
enum OutputFormat {
    /// Human friendly output
    #[default]
    Pretty,
    /// Machine readable output
    Json,
}

#[derive(Parser, Debug)]
#[command(
    name = "ollama-rename",
    version,
    about = "Interactive, safe model renamer for Ollama",
    long_about = "Copy or move Ollama models with safety rails.\n\
\n\
Highlights:\n\
- Interactive picker with clear prompts and running-model warnings.\n\
- Non-interactive rename with JSON output for scripts.\n\
- Built-in list + doctor commands for quick checks.\n\
- Nothing is deleted unless you confirm (or pass --yes)."
)]
struct Cli {
    /// Set Ollama base URL (e.g. http://127.0.0.1:11434). Falls back to OLLAMA_HOST or http://127.0.0.1:11434.
    #[arg(long)]
    host: Option<String>,

    /// Use the Ollama CLI as a fallback if API calls fail (runs `ollama cp`/`ollama rm`)
    #[arg(long, action=ArgAction::SetTrue)]
    use_cli_fallback: bool,

    /// Auto-confirm prompts (use with caution; skips interactive confirmations)
    #[arg(long, action=ArgAction::SetTrue, global = true)]
    yes: bool,

    /// Choose output format for non-interactive commands
    #[arg(long, value_enum, default_value_t = OutputFormat::Pretty, global = true)]
    output: OutputFormat,

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
    /// List local models (optionally only those currently running)
    List {
        /// Show only models that are currently loaded/running
        #[arg(long, action=ArgAction::SetTrue)]
        running_only: bool,
    },
    /// Diagnose connectivity and service health
    Doctor,
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

#[derive(Deserialize, Debug)]
struct VersionResponse {
    version: Option<String>,
}

#[derive(Serialize, Debug)]
struct RenameReport {
    host: String,
    source: String,
    destination: String,
    copied: bool,
    deleted_original: bool,
    overwrote_destination: bool,
    dry_run: bool,
}

#[derive(Copy, Clone)]
struct NonInteractiveOpts {
    delete_original: bool,
    force: bool,
    dry_run: bool,
    use_cli_fallback: bool,
    overwrite: bool,
    output: OutputFormat,
}

#[derive(Serialize, Debug)]
struct ModelSummary {
    name: String,
    size: Option<String>,
    modified_at: Option<String>,
    running: bool,
}

#[derive(Serialize, Debug)]
struct DoctorReport {
    host: String,
    reachable: bool,
    version: Option<String>,
    model_count: usize,
    running_count: usize,
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

    match cli.command {
        Some(Cmd::Rename {
            from,
            to,
            delete_original,
            force,
            dry_run,
            overwrite,
        }) => {
            let opts = NonInteractiveOpts {
                delete_original,
                force,
                dry_run,
                use_cli_fallback: cli.use_cli_fallback,
                overwrite,
                output: cli.output,
            };
            run_non_interactive(&client, &base, &from, &to, opts)
        }
        Some(Cmd::List { running_only }) => run_list(&client, &base, running_only, cli.output),
        Some(Cmd::Doctor) => run_doctor(&client, &base, cli.output),
        None => run_interactive(&client, &base, cli.use_cli_fallback, cli.yes),
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

fn run_interactive(
    client: &Client,
    base: &str,
    use_cli_fallback: bool,
    auto_confirm: bool,
) -> Result<()> {
    let theme = ColorfulTheme::default();
    let version = fetch_version(client, base).ok().flatten();
    println!(
        "{}",
        style("========== Ollama Rename ==========").bold().cyan()
    );
    println!(
        "{}",
        style("Ollama model renamer (safe copy + optional delete)").bold()
    );
    println!(
        "{}",
        style(format!(
            "Target: {}{}",
            base,
            version
                .as_ref()
                .map(|v| format!(" (Ollama {})", v))
                .unwrap_or_default()
        ))
        .dim()
    );
    println!(
        "{}",
        style("Nothing will be deleted unless you say so. Cancel any time with Ctrl+C.").dim()
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

    let running_models = match running_model_names(client, base) {
        Ok(set) => set,
        Err(e) => {
            eprintln!(
                "{}",
                style(format!(
                    "Warning: could not read running models from /api/ps ({})",
                    e
                ))
                .yellow()
            );
            HashSet::new()
        }
    };

    // Show list with fuzzy select
    let items: Vec<String> = models
        .iter()
        .map(|m| format_model(m, running_models.contains(&m.name)))
        .collect();
    let idx = FuzzySelect::with_theme(&theme)
        .with_prompt("Step 1: Pick the model you want to copy/rename")
        .items(&items)
        .default(0)
        .interact()?;

    let chosen = &models[idx];
    println!("Selected: {}", style(&chosen.name).green());

    let suggested = suggest_simple_name(&chosen.name);
    let new_name: String = Input::with_theme(&theme)
        .with_prompt("Step 2: Choose the new name (letters/numbers . _ - / : )")
        .with_initial_text(&suggested)
        .validate_with(|input: &String| validate_model_name(input))
        .interact_text()?;
    let new_name = new_name.trim().to_string();

    if new_name == chosen.name {
        bail!("Destination name equals source; nothing to do.");
    }

    // Prevent accidental overwrite
    if model_exists(client, base, &new_name)? {
        let overwrite = if auto_confirm {
            println!(
                "{}",
                style("Destination exists; proceeding due to --yes.").yellow()
            );
            true
        } else {
            Confirm::with_theme(&theme)
                .with_prompt(format!(
                    "'{}' already exists. Overwrite (delete it first)?",
                    &new_name
                ))
                .default(false)
                .interact()?
        };
        if !overwrite {
            println!("{}", style("Aborted (destination exists).").yellow());
            return Ok(());
        }
        delete_model(client, base, &new_name, use_cli_fallback)
            .with_context(|| format!("Failed to delete existing destination '{}'", &new_name))?;
    }

    println!(
        "{} {} -> {}",
        style("Step 3: Copying").cyan().bold(),
        style(&chosen.name).yellow(),
        style(&new_name).yellow()
    );
    copy_model(client, base, &chosen.name, &new_name, use_cli_fallback)
        .with_context(|| format!("Copy failed from '{}' to '{}'", &chosen.name, &new_name))?;
    println!("{}", style("Copy OK.").green());

    // Offer delete
    let delete = if auto_confirm {
        println!(
            "{}",
            style("Auto-confirming delete of original due to --yes.").yellow()
        );
        true
    } else {
        Confirm::with_theme(&theme)
            .with_prompt(format!(
                "Step 4: Delete original '{}' (makes this a move)?",
                &chosen.name
            ))
            .default(false)
            .interact()?
    };

    if delete {
        if model_is_running(client, base, &chosen.name).unwrap_or(false) {
            let proceed = if auto_confirm {
                println!(
                    "{}",
                    style("Model appears loaded; proceeding due to --yes.").yellow()
                );
                true
            } else {
                Confirm::with_theme(&theme)
                    .with_prompt(
                        "Model seems loaded (`ollama ps`). Stop it first. Proceed with delete anyway?",
                    )
                    .default(false)
                    .interact()?
            };
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
    println!(
        "{}",
        style("Tip: rerun with --output json for scripting, or --yes to auto-confirm.").dim()
    );
    Ok(())
}

fn run_non_interactive(
    client: &Client,
    base: &str,
    from: &str,
    to: &str,
    opts: NonInteractiveOpts,
) -> Result<()> {
    // Normalize/trim
    let to = to.trim();
    let from = from.trim();
    validate_model_name(to).map_err(|e| anyhow::anyhow!(e))?;
    validate_model_name(from).map_err(|e| anyhow::anyhow!(e))?;

    let mut report = RenameReport {
        host: base.to_string(),
        source: from.to_string(),
        destination: to.to_string(),
        copied: false,
        deleted_original: false,
        overwrote_destination: false,
        dry_run: opts.dry_run,
    };

    let models = list_models(client, base)?;
    let source_exists = models.iter().any(|m| m.name == from);
    if !source_exists {
        bail!(
            "Source model '{}' was not found on this Ollama instance.",
            from
        );
    }

    let dest_exists = models.iter().any(|m| m.name == to);

    if opts.dry_run {
        println!("[dry-run] Would copy '{}' -> '{}'", from, to);
        if opts.delete_original {
            println!("[dry-run] Would delete original '{}'", from);
        }
        if opts.output == OutputFormat::Json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        return Ok(());
    }

    if dest_exists && opts.overwrite {
        delete_model(client, base, to, opts.use_cli_fallback)
            .with_context(|| format!("Failed to delete existing destination '{}'", to))?;
        report.overwrote_destination = true;
    } else if dest_exists {
        bail!(
            "Destination '{}' already exists. Use --overwrite to replace it.",
            to
        );
    }

    copy_model(client, base, from, to, opts.use_cli_fallback)
        .with_context(|| format!("Copy failed from '{}' to '{}'", from, to))?;
    println!("{}", style("Copy OK.").green());
    report.copied = true;

    if opts.delete_original {
        if !opts.force && model_is_running(client, base, from).unwrap_or(false) {
            bail!("Model appears loaded (via /api/ps). Use --force to attempt delete anyway, or stop it first.");
        }
        delete_model(client, base, from, opts.use_cli_fallback)
            .with_context(|| format!("Failed to delete '{}'", from))?;
        println!("{}", style("Deleted original.").green());
        report.deleted_original = true;
    }

    if opts.output == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    Ok(())
}

fn run_list(client: &Client, base: &str, running_only: bool, output: OutputFormat) -> Result<()> {
    let mut models = list_models(client, base)?;
    models.sort_by(|a, b| {
        b.modified_at
            .cmp(&a.modified_at)
            .then_with(|| a.name.cmp(&b.name))
    });

    let running = running_model_names(client, base)?;
    let summaries = summarize_models(&models, &running, running_only);

    if output == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
        return Ok(());
    }

    if summaries.is_empty() {
        println!("{}", style("No matching models found.").yellow());
        return Ok(());
    }

    println!(
        "{}",
        style(format!(
            "Models on {}{}",
            base,
            if running_only { " (running only)" } else { "" }
        ))
        .bold()
    );
    for (idx, summary) in summaries.iter().enumerate() {
        println!("{:>3}. {}", idx + 1, format_model_summary(summary));
    }
    Ok(())
}

fn run_doctor(client: &Client, base: &str, output: OutputFormat) -> Result<()> {
    let reachable = is_ollama_api_running(client, base);
    let version = fetch_version(client, base).ok().flatten();
    let models = if reachable {
        list_models(client, base)?
    } else {
        Vec::new()
    };
    let running = if reachable {
        running_model_names(client, base)?
    } else {
        HashSet::new()
    };
    let report = DoctorReport {
        host: base.to_string(),
        reachable,
        version,
        model_count: models.len(),
        running_count: running.len(),
    };

    if output == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("{}", style("Ollama doctor").bold());
    println!("Host: {}", base);
    if !report.reachable {
        println!("{}", style("API not reachable right now.").red());
        return Ok(());
    }
    if let Some(v) = &report.version {
        println!("Version: {}", v);
    } else {
        println!("Version: (unknown)");
    }
    println!("Models: {}", report.model_count);
    println!("Running: {}", report.running_count);

    if report.model_count == 0 {
        println!(
            "{}",
            style("No models found. Use `ollama pull ...` to add one.").yellow()
        );
    }
    Ok(())
}

fn validate_model_name(s: &str) -> std::result::Result<(), String> {
    // Validate allowed characters and structure manually to avoid regex overhead.
    fn is_allowed(c: char) -> bool {
        c.is_ascii_alphanumeric() || ".-_".contains(c)
    }

    if s.is_empty() {
        return Err("Invalid name: empty".into());
    }

    // Split path and optional tag
    let (path, tag) = match s.split_once(':') {
        Some((p, t)) => (p, Some(t)),
        None => (s, None),
    };

    if path.is_empty() || path.contains("//") {
        return Err("Invalid name. Use letters, numbers, . _ - / and optional :tag".into());
    }

    for segment in path.split('/') {
        if segment.is_empty() || !segment.chars().all(is_allowed) {
            return Err("Invalid name. Use letters, numbers, . _ - / and optional :tag".into());
        }
    }

    if let Some(tag) = tag {
        if tag.is_empty() || !tag.chars().all(is_allowed) {
            return Err("Invalid name. Use letters, numbers, . _ - / and optional :tag".into());
        }
    }

    Ok(())
}

fn suggest_simple_name(full: &str) -> String {
    // 1) strip tag
    let before_tag = full.split(':').next().unwrap_or(full);
    // 2) take last path segment
    let last = before_tag.split('/').next_back().unwrap_or(before_tag);
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

fn fetch_version(client: &Client, base: &str) -> Result<Option<String>> {
    let url = api_url(base, "/api/version");
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .context("GET /api/version failed")?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let vr: VersionResponse = resp.json().context("Decode /api/version JSON")?;
    Ok(vr.version)
}

fn running_model_names(client: &Client, base: &str) -> Result<HashSet<String>> {
    let url = api_url(base, "/api/ps");
    let resp = client.get(&url).send().context("GET /api/ps failed")?;
    if !resp.status().is_success() {
        return Ok(HashSet::new());
    }
    let pr: PsResponse = resp.json().context("Decode /api/ps JSON")?;
    let mut set = HashSet::new();
    for name in pr
        .models
        .unwrap_or_default()
        .into_iter()
        .filter_map(|m| m.name)
    {
        set.insert(name);
    }
    Ok(set)
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
            .args(["/C", "start", "ollama", "serve"])
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
    let running = running_model_names(client, base)?;
    Ok(running.contains(name))
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

fn format_model(m: &ModelInfo, running: bool) -> String {
    let mut s = m.name.clone();
    if let Some(szv) = &m.size {
        if let Some(sz_str) = fmt_size_value(szv) {
            s.push_str(&format!("  ({})", sz_str));
        }
    }
    if let Some(modified) = &m.modified_at {
        s.push_str(&format!("  (modified {})", modified));
    }
    if running {
        s.push_str("  [running]");
    }
    s
}

fn summarize_models(
    models: &[ModelInfo],
    running: &HashSet<String>,
    running_only: bool,
) -> Vec<ModelSummary> {
    models
        .iter()
        .filter(|m| !running_only || running.contains(&m.name))
        .map(|m| ModelSummary {
            name: m.name.clone(),
            size: m.size.as_ref().and_then(fmt_size_value),
            modified_at: m.modified_at.clone(),
            running: running.contains(&m.name),
        })
        .collect()
}

fn format_model_summary(m: &ModelSummary) -> String {
    let mut s = m.name.clone();
    if let Some(sz) = &m.size {
        s.push_str(&format!("  ({})", sz));
    }
    if let Some(modified) = &m.modified_at {
        s.push_str(&format!("  (modified {})", modified));
    }
    if m.running {
        s.push_str("  [running]");
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

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use std::env;

    /// RAII helper to temporarily set an env var and restore it on drop.
    struct EnvOverride {
        key: String,
        original: Option<String>,
    }

    impl EnvOverride {
        fn new(key: &str, value: &str) -> Self {
            let original = env::var(key).ok();
            env::set_var(key, value);
            EnvOverride {
                key: key.to_string(),
                original,
            }
        }
    }

    impl Drop for EnvOverride {
        fn drop(&mut self) {
            match &self.original {
                Some(v) => env::set_var(&self.key, v),
                None => env::remove_var(&self.key),
            }
        }
    }

    #[test]
    fn validate_model_name_allows_safe_patterns() {
        assert!(validate_model_name("llama3").is_ok());
        assert!(validate_model_name("hf.co/user/model:latest").is_ok());
        assert!(validate_model_name("myspace/sub/model_Q4_K").is_ok());
    }

    #[test]
    fn validate_model_name_rejects_bad_patterns() {
        assert!(validate_model_name("bad name").is_err());
        assert!(validate_model_name(":starts-with-colon").is_err());
        assert!(validate_model_name("double//slash").is_err());
    }

    #[test]
    fn suggest_simple_name_strips_noise() {
        let suggested = suggest_simple_name("hf.co/a/b/CoolModel-Q4_K_M-GGUF:Q4_K_M");
        assert_eq!(suggested, "CoolModel");
    }

    #[test]
    fn pick_base_url_prefers_cli_value() {
        let _guard = EnvOverride::new("OLLAMA_HOST", "10.0.0.2:11434");
        assert_eq!(pick_base_url(Some("1.2.3.4:9999")), "http://1.2.3.4:9999");
    }

    #[test]
    fn run_non_interactive_overwrites_and_deletes_via_api() {
        let server = MockServer::start();

        let _tags = server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .json_body(json!({"models":[{"name":"src"},{"name":"dest"}]}));
        });

        let copy = server.mock(|when, then| {
            when.method(POST).path("/api/copy");
            then.status(200);
        });

        let delete = server.mock(|when, then| {
            when.method(DELETE).path("/api/delete");
            then.status(200);
        });

        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();

        let opts = NonInteractiveOpts {
            delete_original: true,
            force: true,
            dry_run: false,
            use_cli_fallback: false,
            overwrite: true,
            output: OutputFormat::Json,
        };

        run_non_interactive(&client, server.base_url().as_str(), "src", "dest", opts).unwrap();

        copy.assert();
        delete.assert_hits(2);
    }

    #[test]
    fn running_model_names_reads_ps() {
        let server = MockServer::start();

        let _ps = server.mock(|when, then| {
            when.method(GET).path("/api/ps");
            then.status(200).json_body(json!({
                "models": [
                    {"name":"one"},
                    {"name":"two"}
                ]
            }));
        });

        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let set = running_model_names(&client, server.base_url().as_str()).unwrap();
        assert!(set.contains("one"));
        assert!(set.contains("two"));
    }
}
