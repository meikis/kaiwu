# Kaiwu Installer for Windows
# Usage: irm https://raw.githubusercontent.com/val1813/kaiwu/main/install.ps1 | iex

$ErrorActionPreference = "Stop"
$Repo = "val1813/kaiwu"
$BinName = "kaiwu.exe"
$InstallDir = "$env:USERPROFILE\.kaiwu\bin"

Write-Host "Kaiwu Installer" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan
Write-Host ""

# Detect architecture
$Arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "386" }
Write-Host "Detected: windows/$Arch"

# Get latest release
Write-Host "Fetching latest release..."
try {
    $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
    $Tag = $Release.tag_name
} catch {
    Write-Host "Error: could not fetch latest release." -ForegroundColor Red
    Write-Host "Check https://github.com/$Repo/releases"
    exit 1
}
Write-Host "Latest version: $Tag"

# Build download URL
$Asset = "kaiwu-windows-${Arch}.zip"
$Url = "https://github.com/$Repo/releases/download/$Tag/$Asset"

# Download
$TmpDir = Join-Path $env:TEMP "kaiwu-install-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
$ZipPath = Join-Path $TmpDir "kaiwu.zip"

Write-Host "Downloading $Url..."
try {
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing
} catch {
    # Fallback: try raw .exe
    $Url = "https://github.com/$Repo/releases/download/$Tag/kaiwu.exe"
    Write-Host "Trying raw binary: $Url..."
    try {
        Invoke-WebRequest -Uri $Url -OutFile (Join-Path $TmpDir $BinName) -UseBasicParsing
    } catch {
        Write-Host "Error: download failed." -ForegroundColor Red
        Write-Host "Check available assets at: https://github.com/$Repo/releases/tag/$Tag"
        Remove-Item -Recurse -Force $TmpDir
        exit 1
    }
}

# Extract if zip
if (Test-Path $ZipPath) {
    Expand-Archive -Path $ZipPath -DestinationPath $TmpDir -Force
}

# Create install directory
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
}

# Find and move binary
$ExePath = Get-ChildItem -Path $TmpDir -Filter $BinName -Recurse | Select-Object -First 1
if (-not $ExePath) {
    Write-Host "Error: $BinName not found in download." -ForegroundColor Red
    Remove-Item -Recurse -Force $TmpDir
    exit 1
}
Copy-Item -Path $ExePath.FullName -Destination (Join-Path $InstallDir $BinName) -Force

# Clean up
Remove-Item -Recurse -Force $TmpDir

# Add to PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Write-Host "Added $InstallDir to user PATH." -ForegroundColor Green
}

# Verify
Write-Host ""
$KaiwuPath = Join-Path $InstallDir $BinName
if (Test-Path $KaiwuPath) {
    Write-Host "Kaiwu installed successfully!" -ForegroundColor Green
    Write-Host ""
    & $KaiwuPath version
    Write-Host ""
    Write-Host "Get started:" -ForegroundColor Cyan
    Write-Host "  kaiwu run Qwen3-30B-A3B"
    Write-Host ""
    Write-Host "Note: restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
} else {
    Write-Host "Error: installation failed." -ForegroundColor Red
    exit 1
}
