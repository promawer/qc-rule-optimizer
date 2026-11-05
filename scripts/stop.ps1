# 停止由 dev.ps1 启动的服务。
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root '.run\app.pid'

if (-not (Test-Path -LiteralPath $pidFile)) {
  Write-Host '没有正在运行的服务（.run\app.pid 不存在）。'
  exit 0
}

$processId = 0
$raw = (Get-Content -LiteralPath $pidFile -Raw).Trim()
if (-not [int]::TryParse($raw, [ref]$processId)) {
  Remove-Item -LiteralPath $pidFile -Force
  Write-Host 'PID 文件无效，已清理。'
  exit 0
}

$commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue).CommandLine
if (-not $commandLine -or $commandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
  Write-Host '进程已退出或 PID 不属于本项目，未发送终止信号。'
} else {
  & taskkill.exe /PID $processId /T /F *> $null
  Write-Host "已停止服务（PID $processId）。"
}
Remove-Item -LiteralPath $pidFile -Force
