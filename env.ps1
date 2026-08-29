# Activate the project environment.  Run from the repo root:  . .\env.ps1
#
# Portable: by default it expects  venv\  models\  whisper\  ollama\  data\
# to live inside the repo folder. If yours live elsewhere (e.g. a different
# drive because C: is full), create env.local.ps1 next to this file — see
# env.local.ps1.example — and set $DataRoot / $OllamaExe there.
#
# NOTE (Windows): if ". .\env.ps1" errors with "running scripts is disabled",
# run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$RepoRoot  = $PSScriptRoot
$DataRoot  = $PSScriptRoot                     # holds venv\ models\ whisper\ ollama\ (heavy — env.local.ps1 moves this off C:)
$PipelineDataRoot = $RepoRoot                  # the corpus + Chroma index — kept WITH the code so the editor sees them
$OllamaExe = "ollama"                          # or a full path if not on PATH

if (Test-Path (Join-Path $PSScriptRoot "env.local.ps1")) {
    . (Join-Path $PSScriptRoot "env.local.ps1")
}

$venv = Join-Path $DataRoot "venv"

# --- activate the venv (manually — independent of its activate scripts) ----
$env:VIRTUAL_ENV = $venv
$env:PATH = (Join-Path $venv "Scripts") + ";" + $env:PATH
Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue

# --- model / weight caches -----------------------------------------------
$env:HF_HOME           = Join-Path $DataRoot "hf"
$env:TORCH_HOME        = Join-Path $DataRoot "models\torch"
$env:RAG_MODEL_CACHE   = Join-Path $DataRoot "models"       # ImageBind checkpoint
$env:RAG_WHISPER_CACHE = Join-Path $DataRoot "whisper"      # Whisper model files
$env:OLLAMA_MODELS     = Join-Path $DataRoot "ollama"       # Ollama pulled models
$env:PIP_CACHE_DIR     = Join-Path $DataRoot "pipcache"
$env:TEMP              = Join-Path $DataRoot "tmp"
$env:TMP               = Join-Path $DataRoot "tmp"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

# --- pipeline data (lives with the code, not on $DataRoot) --------------
$env:RAG_RAW_DATA_DIR = Join-Path $PipelineDataRoot "data\raw"
$env:RAG_CHROMA_DIR   = Join-Path $PipelineDataRoot "data\chroma"
$env:RAG_MANIFEST     = Join-Path $PipelineDataRoot "data\corpus_manifest.csv"

# --- Ollama: ensure a server is up, pointed at the local model dir --------
if ((Get-Command $OllamaExe -ErrorAction SilentlyContinue) -and
    -not (Get-Process -Name ollama -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 3
}

Write-Host "env ready:  python -> $((Get-Command python -ErrorAction SilentlyContinue).Source)" -ForegroundColor Green
Write-Host "repo: $RepoRoot   data/venv/models: $DataRoot" -ForegroundColor DarkGray
