# 离线环境自检：运行后端单元测试和字节码编译检查；有 Node 时再检查前端 JS。
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$bundled = Join-Path $root 'runtime\python\python.exe'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

if (Test-Path -LiteralPath $bundled) {
  $python = $bundled
} elseif (Test-Path -LiteralPath $venvPython) {
  $python = $venvPython
} else {
  throw '未找到 Python 运行时，请先运行 start.cmd 初始化环境。'
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $root
try {
  Write-Host '=== 单元测试 ===' -ForegroundColor Cyan
  & $python (Join-Path $root 'scripts\run_tests.py')
  if ($LASTEXITCODE -ne 0) { throw '单元测试失败' }

  Write-Host ''
  Write-Host '=== 字节码编译检查 ===' -ForegroundColor Cyan
  & $python -m compileall -q app
  if ($LASTEXITCODE -ne 0) { throw '字节码编译失败' }
} finally {
  Pop-Location
}

$node = Get-Command node -ErrorAction SilentlyContinue
if ($node) {
  Write-Host ''
  Write-Host '=== 前端 JS 语法检查 ===' -ForegroundColor Cyan
  & $node.Source --check (Join-Path $root 'static\app.js')
  if ($LASTEXITCODE -ne 0) { throw '前端 JS 语法检查失败' }
}

Write-Host ''
Write-Host '全部检查通过。' -ForegroundColor Green
