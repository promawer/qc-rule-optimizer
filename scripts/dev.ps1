# 本地启动（Windows / PowerShell）：校验环境后交给 scripts\launch.py 启动。
[CmdletBinding()]
param(
  [switch]$NoBrowser,
  [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$port = 8090

function Test-PortFree([int]$Port) {
  $listener = $null
  try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $listener.Start()
    return $true
  } catch {
    return $false
  } finally {
    if ($listener) { $listener.Stop() }
  }
}

# 服务已在运行就直接返回，避免重复启动。
try {
  $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 2
  if ($response.StatusCode -eq 200) {
    Write-Host "服务已在运行：http://127.0.0.1:$port" -ForegroundColor Yellow
    exit 0
  }
} catch { }

if (-not (Test-PortFree $port)) {
  throw "端口 $port 已被占用，请先停止占用该端口的程序。"
}

if (-not (Test-Path -LiteralPath $python)) {
  Write-Host '创建 Python 虚拟环境…'
  & py -3 -m venv $venv
  if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败，请确认已安装 Python 3.9+。' }
}

if (-not $SkipInstall) {
  $marker = Join-Path $venv '.deps-installed'
  if (-not (Test-Path -LiteralPath $marker)) {
    Write-Host '安装 Python 依赖…'
    $env:PIP_CACHE_DIR = 'E:\qc-runtimes\pip-cache'
    & $python -m pip install --quiet -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败，请检查网络或 pip 源。' }
    New-Item -ItemType File -Path $marker | Out-Null
  }
}

$arguments = @((Join-Path $root 'scripts\launch.py'))
if ($NoBrowser) { $arguments += '--no-browser' }
& $python @arguments
exit $LASTEXITCODE
