# 运行后端单元测试、字节码编译检查和前端 JS 语法检查。
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  throw '未找到 .venv，请先运行 scripts\dev.ps1 初始化环境。'
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $root
try {
  Write-Host '=== 单元测试 ===' -ForegroundColor Cyan
  & $python -m unittest discover -s tests -v
  if ($LASTEXITCODE -ne 0) { throw '单元测试失败' }

  Write-Host ''
  Write-Host '=== 字节码编译检查 ===' -ForegroundColor Cyan
  & $python -m compileall -q app
  if ($LASTEXITCODE -ne 0) { throw '字节码编译失败' }
} finally {
  Pop-Location
}

$node = 'E:\qc-runtimes\node\node.exe'
if (Test-Path -LiteralPath $node) {
  Write-Host ''
  Write-Host '=== 前端 JS 语法检查 ===' -ForegroundColor Cyan
  & $node --check (Join-Path $root 'static\app.js')
  if ($LASTEXITCODE -ne 0) { throw '前端 JS 语法检查失败' }
}

Write-Host ''
Write-Host '全部检查通过。' -ForegroundColor Green
