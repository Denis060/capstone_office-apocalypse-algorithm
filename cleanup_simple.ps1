# Simple Project Cleanup Script for GitHub Submission
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "OFFICE APOCALYPSE ALGORITHM - PROJECT CLEANUP" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Get-Location
Write-Host "Project Root: $projectRoot" -ForegroundColor Yellow
Write-Host ""

# 1. Remove __pycache__ directories
Write-Host "1. Removing __pycache__ directories..." -ForegroundColor Green
$pycache = Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\venv\*" }
$count = 0
foreach ($dir in $pycache) {
    Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $count++
}
Write-Host "   Removed $count __pycache__ directories" -ForegroundColor White
Write-Host ""

# 2. Check for duplicate venvs
Write-Host "2. Checking for duplicate virtual environments..." -ForegroundColor Green
$venvExists = Test-Path ".\venv"
$dotVenvExists = Test-Path ".\.venv"

if ($venvExists -and $dotVenvExists) {
    Write-Host "   WARNING: Both venv and .venv exist!" -ForegroundColor Yellow
    Write-Host "   Active: .venv (currently used)" -ForegroundColor Yellow
    Write-Host "   You can manually delete 'venv' folder to save space" -ForegroundColor Yellow
}
elseif ($venvExists) {
    Write-Host "   Only venv directory exists - OK" -ForegroundColor White
}
elseif ($dotVenvExists) {
    Write-Host "   Only .venv directory exists - OK" -ForegroundColor White
}
Write-Host ""

# 3. Remove Jupyter checkpoints
Write-Host "3. Removing Jupyter notebook checkpoints..." -ForegroundColor Green
$checkpoints = Get-ChildItem -Path . -Recurse -Directory -Filter ".ipynb_checkpoints" -ErrorAction SilentlyContinue
$count = 0
foreach ($checkpoint in $checkpoints) {
    Remove-Item -Path $checkpoint.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $count++
}
Write-Host "   Removed $count checkpoint directories" -ForegroundColor White
Write-Host ""

# 4. Remove temporary files
Write-Host "4. Removing temporary files..." -ForegroundColor Green
$tempPatterns = @("*.pyc", "*.swp", "*.swo", ".DS_Store", "Thumbs.db")
$count = 0
foreach ($pattern in $tempPatterns) {
    $files = Get-ChildItem -Path . -Recurse -Filter $pattern -File -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\venv\*" }
    foreach ($file in $files) {
        Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
        $count++
    }
}
Write-Host "   Removed $count temporary files" -ForegroundColor White
Write-Host ""

# 5. Check for large files
Write-Host "5. Checking for large files (>100MB)..." -ForegroundColor Green
$largeFiles = Get-ChildItem -Path . -Recurse -File -ErrorAction SilentlyContinue |
              Where-Object {
                  $_.Length -gt 100MB -and
                  $_.FullName -notlike "*\.venv\*" -and
                  $_.FullName -notlike "*\venv\*" -and
                  $_.FullName -notlike "*\.git\*"
              }

if ($largeFiles) {
    Write-Host "   WARNING: Large files found:" -ForegroundColor Yellow
    foreach ($file in $largeFiles) {
        $sizeMB = [math]::Round($file.Length/1MB, 2)
        Write-Host "   - $($file.Name): $sizeMB MB" -ForegroundColor Yellow
    }
}
else {
    Write-Host "   No large files found" -ForegroundColor White
}
Write-Host ""

# 6. Verify .gitignore
Write-Host "6. Verifying .gitignore configuration..." -ForegroundColor Green
if (Test-Path ".\.gitignore") {
    Write-Host "   .gitignore exists" -ForegroundColor White
}
else {
    Write-Host "   WARNING: .gitignore NOT FOUND!" -ForegroundColor Yellow
}
Write-Host ""

# 7. Check Git status
Write-Host "7. Checking Git repository status..." -ForegroundColor Green
try {
    $gitStatus = git status --short 2>&1
    if ($LASTEXITCODE -eq 0) {
        $uncommittedLines = ($gitStatus | Measure-Object).Count
        if ($uncommittedLines -gt 0) {
            Write-Host "   You have uncommitted changes" -ForegroundColor Yellow
            Write-Host "   Run 'git status' to see details" -ForegroundColor Yellow
        }
        else {
            Write-Host "   Working directory is clean" -ForegroundColor White
        }
    }
}
catch {
    Write-Host "   Git status unavailable" -ForegroundColor Gray
}
Write-Host ""

# Summary
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE!" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEPS FOR GITHUB SUBMISSION:" -ForegroundColor Yellow
Write-Host "  1. git status" -ForegroundColor White
Write-Host "  2. git add ." -ForegroundColor White
Write-Host '  3. git commit -m "Final project cleanup for submission"' -ForegroundColor White
Write-Host "  4. git push origin main" -ForegroundColor White
Write-Host "  5. Verify on GitHub website" -ForegroundColor White
Write-Host ""
Write-Host "Project is ready for submission!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan
