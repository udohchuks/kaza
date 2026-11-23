# Render assets/src/<name>.html to assets/<name>.png (2x) and assets/<name>.pdf.
# Usage (from the repo root):  powershell -File assets/src/render.ps1 summary model pipeline
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Names)

$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path $edge)) { $edge = "C:\Program Files\Google\Chrome\Application\chrome.exe" }

$src = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Split-Path -Parent $src
$sizes = @{ summary = "1080,960"; model = "1080,1440"; pipeline = "1080,1060" }

foreach ($n in $Names) {
    $html = Join-Path $src "$n.html"
    $url = "file:///" + ($html -replace "\\", "/")
    $png = Join-Path $out "$n.png"
    $pdf = Join-Path $out "$n.pdf"
    & $edge --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 `
        --virtual-time-budget=8000 --window-size=$($sizes[$n]) --screenshot="$png" $url | Out-Null
    & $edge --headless=new --disable-gpu --virtual-time-budget=8000 --no-pdf-header-footer `
        --print-to-pdf="$pdf" $url | Out-Null
    Write-Output "$n -> $png, $pdf"
}
