<#
  launch.ps1 -- open a lecture notebook in marimo.
  Usage:
    ./launch.ps1 0b      # opens notebooks/0b_execution_model.py
    ./launch.ps1         # opens the home/index notebook
#>
param([string]$Id)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$nbdir = Join-Path $root "notebooks"

if ([string]::IsNullOrWhiteSpace($Id)) {
    $target = Join-Path $nbdir "home.py"
} else {
    $match = Get-ChildItem -Path $nbdir -Filter "$Id*.py" | Select-Object -First 1
    if ($null -eq $match) {
        Write-Error "No notebook matches '$Id' in $nbdir"
    }
    $target = $match.FullName
}

Write-Host "marimo edit $target"
marimo edit $target
