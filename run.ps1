# One-step launcher: activates the environment, then runs a target.
#   .\run.ps1              -> RAG_OS desktop app (starts the backend if needed)
#   .\run.ps1 server       -> just the backend (FastAPI, 127.0.0.1:8077)
#   .\run.ps1 web          -> the Gradio web UI (in-process, fallback)
#   .\run.ps1 ask "..."    -> ask one question from the command line
#   .\run.ps1 eval         -> retrieval evaluation
#   .\run.ps1 stats        -> Chroma collection stats
#   .\run.ps1 ingest       -> re-index everything under data/raw (--reset)
#   .\run.ps1 ami          -> download the AMI meeting corpus + build the manifest
#   .\run.ps1 assets       -> (legacy) regenerate the old synthetic image/audio corpus
#   .\run.ps1 stop         -> kill the backend + stale servers holding the GPU

$cmd = if ($args.Count -ge 1) { $args[0] } else { "app" }

if ($cmd -eq "stop") {
    Get-Process python, pythonw, llama-server -EA SilentlyContinue |
        Where-Object { $_.Id -ne $PID } | Stop-Process -Force -EA SilentlyContinue
    Get-Process ollama -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep 2
    Write-Host "stopped RAG processes; GPU:" -ForegroundColor Yellow
    try { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader } catch {}
    return
}

. "$PSScriptRoot\env.ps1"

switch ($cmd) {
    "app"     { python -m desktop.main }
    "desktop" { python -m desktop.main }
    "server"  { python -m backend.server }
    "web"     { python "$PSScriptRoot\web\app.py" }
    "eval"    { python "$PSScriptRoot\scripts\evaluate.py" }
    "stats"   { python -m rag.ingest.corpus --stats }
    "ingest"  { python -m rag.ingest.corpus --src (Join-Path $env:RAG_RAW_DATA_DIR "ami") --reset }
    "ami"     { python -m rag.ingest.manifest @($args | Select-Object -Skip 1) }
    "assets"  { python "$PSScriptRoot\scripts\make_demo_assets.py" }
    "ask"     {
        if ($args.Count -lt 2) { Write-Host 'Usage: .\run.ps1 ask "your question"'; break }
        python -m rag.generation.answer $args[1]
    }
    default   { python -m rag.generation.answer $cmd }
}
