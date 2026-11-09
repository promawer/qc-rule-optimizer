# 内网离线启动：优先使用包内自带的 Python 3.8 运行时；
# 没有自带运行时时，回退到本机 Python 3.8 + 本地 wheelhouse 安装。
[CmdletBinding()]
param(
  [switch]$NoBrowser,
  [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
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

function Find-Python38 {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    & cmd.exe /c "`"$($py.Source)`" -3.8 -c `"import sys`" >nul 2>nul"
    if ($LASTEXITCODE -eq 0) {
      return [pscustomobject]@{ Exe = $py.Source; Args = @('-3.8') }
    }
  }

  foreach ($name in 'python', 'python3') {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $command) { continue }
    $expression = "import sys; print('%d.%d' % sys.version_info[:2])"
    $version = (& cmd.exe /c "`"$($command.Source)`" -c `"$expression`" 2>nul").Trim()
    if ($version -eq '3.8') {
      return [pscustomobject]@{ Exe = $command.Source; Args = @() }
    }
  }

  throw '未找到 Python 3.8，且离线包缺少 runtime\python。请确认压缩包解压完整。'
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
  throw "端口 $port 已被占用，请先关闭占用该端口的程序，或执行 stop.cmd。"
}

$bundled = Join-Path $root 'runtime\python\python.exe'
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

if (Test-Path -LiteralPath $bundled) {
  $python = $bundled
  Write-Host '使用包内 Python 3.8 运行时。'
  & $python -c "import fastapi, uvicorn, pydantic, httpx, openpyxl, multipart" 2>$null
  if ($LASTEXITCODE -ne 0) {
    throw '包内 Python 运行时缺少依赖，请确认 runtime\python\Lib\site-packages 完整。'
  }
} else {
  if (-not (Test-Path -LiteralPath $venvPython)) {
    $system = Find-Python38
    Write-Host '创建 Python 3.8 虚拟环境…'
    $venvArgs = @($system.Args) + @('-m', 'venv', $venv)
    & $system.Exe @venvArgs
    if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败。' }
  }
  $python = $venvPython

  $wheelhouse = Join-Path $root 'wheelhouse'
  $marker = Join-Path $venv '.deps-installed'
  if (-not $SkipInstall -and -not (Test-Path -LiteralPath $marker)) {
    if (-not (Test-Path -LiteralPath $wheelhouse)) {
      throw '缺少 wheelhouse 离线依赖目录，请确认压缩包解压完整。'
    }
    Write-Host '从本地 wheelhouse 安装离线依赖…'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    & $python -m pip install --no-index --find-links $wheelhouse --only-binary=:all: --no-cache-dir -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '离线依赖安装失败，请确认 wheelhouse 完整且 Python 为 3.8 64 位。' }
    New-Item -ItemType File -Path $marker | Out-Null
  }
}

$arguments = @((Join-Path $root 'scripts\launch.py'))
if ($NoBrowser) { $arguments += '--no-browser' }
& $python @arguments
exit $LASTEXITCODE
