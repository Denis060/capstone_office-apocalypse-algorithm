# Project Cleanup Script for GitHub Submission
# Removes unnecessary files and prepares project for submission

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "OFFICE APOCALYPSE ALGORITHM - PROJECT CLEANUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = "C:\Users\pcric\Desktop\capstone_project\office_apocalypse_algorithm_project"
Set-Location $projectRoot

# Track cleanup statistics
$stats = @{
    pycache_removed = 0
    duplicate_venv_found = $false
    temp_files_removed = 0
    large_files_found = @()
}

Write-Host "📁 Current Project Structure:" -ForegroundColor Yellow
Get-ChildItem -Directory | Select-Object Name | Format-Table -AutoSize

Write-Host "`n🔍 Starting cleanup process...`n" -ForegroundColor Green

# 1. Remove all __pycache__ directories
Write-Host "1. Removing __pycache__ directories..." -ForegroundColor Cyan
$pycacheDirs = Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
foreach ($dir in $pycacheDirs) {
    if ($dir.FullName -notlike "*\.venv\*" -and $dir.FullName -notlike "*\venv\*") {
        Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $stats.pycache_removed++
        Write-Host "   ✅ Removed: $($dir.FullName)" -ForegroundColor Green
    }
}
Write-Host "   📊 Removed $($stats.pycache_removed) __pycache__ directories`n" -ForegroundColor Green

# 2. Check for duplicate virtual environments
Write-Host "2. Checking for duplicate virtual environments..." -ForegroundColor Cyan
$venvExists = Test-Path ".\venv"
$dotVenvExists = Test-Path ".\.venv"

if ($venvExists -and $dotVenvExists) {
    Write-Host "   WARNING: Both 'venv' and '.venv' directories exist!" -ForegroundColor Yellow
    Write-Host "   Active environment: .venv (used in terminals)" -ForegroundColor Yellow
    Write-Host "   The 'venv' directory can be safely deleted to save space" -ForegroundColor Yellow
    $stats.duplicate_venv_found = $true
    
    $response = Read-Host "   Do you want to delete the duplicate 'venv' folder? (yes/no)"
    if ($response -eq "yes") {
        Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "   Deleted duplicate 'venv' directory" -ForegroundColor Green
    }
    else {
        Write-Host "   Skipped deletion of 'venv' directory" -ForegroundColor Yellow
    }
}
elseif ($venvExists) {
    Write-Host "   Only 'venv' directory exists - OK" -ForegroundColor Green
}
elseif ($dotVenvExists) {
    Write-Host "   Only '.venv' directory exists - OK" -ForegroundColor Green
}

# 3. Remove .ipynb_checkpoints
Write-Host "3. Removing Jupyter notebook checkpoints..." -ForegroundColor Cyan
$checkpoints = Get-ChildItem -Path . -Recurse -Directory -Filter ".ipynb_checkpoints" -ErrorAction SilentlyContinue
foreach ($checkpoint in $checkpoints) {
    Remove-Item -Path $checkpoint.FullName -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "   ✅ Removed: $($checkpoint.FullName)" -ForegroundColor Green
}
Write-Host "   📊 Removed $($checkpoints.Count) checkpoint directories`n" -ForegroundColor Green

# 4. Remove temporary files
Write-Host "4. Removing temporary files (.pyc, .swp, .swo, .DS_Store)..." -ForegroundColor Cyan
$tempExtensions = @("*.pyc", "*.swp", "*.swo", ".DS_Store", "Thumbs.db")
foreach ($ext in $tempExtensions) {
    $files = Get-ChildItem -Path . -Recurse -Filter $ext -File -ErrorAction SilentlyContinue | 
             Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\venv\*" }
    foreach ($file in $files) {
        Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
        $stats.temp_files_removed++
    }
}
Write-Host "   ✅ Removed $($stats.temp_files_removed) temporary files`n" -ForegroundColor Green

# 5. Check for large files that shouldn't be in GitHub
Write-Host "5. Checking for large files (>100MB)..." -ForegroundColor Cyan
$largeFiles = Get-ChildItem -Path . -Recurse -File -ErrorAction SilentlyContinue | 
              Where-Object { 
                  $_.Length -gt 100MB -and 
                  $_.FullName -notlike "*\.venv\*" -and 
                  $_.FullName -notlike "*\venv\*" -and
                  $_.FullName -notlike "*\.git\*"
              } | Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}

if ($largeFiles) {
    Write-Host "   ⚠️  WARNING: Large files found that may cause GitHub issues:" -ForegroundColor Yellow
    $largeFiles | ForEach-Object {
        Write-Host "   📦 $($_.FullName) - $($_.SizeMB) MB" -ForegroundColor Yellow
        $stats.large_files_found += $_.FullName
    }
} else {
    Write-Host "   No large files found" -ForegroundColor Green
    Write-Host ""
}

# 6. Verify .gitignore is working
Write-Host "`n6. Verifying .gitignore configuration..." -ForegroundColor Cyan
if (Test-Path ".\.gitignore") {
    Write-Host "   ✅ .gitignore exists" -ForegroundColor Green
    $gitignoreContent = Get-Content ".\.gitignore"
    $requiredPatterns = @(".venv", "venv/", "__pycache__", "*.pyc", "data/raw/*.csv", "models/*.pkl")
    $missingPatterns = @()
    
    foreach ($pattern in $requiredPatterns) {
        if ($gitignoreContent -notcontains $pattern -and $gitignoreContent -notmatch [regex]::Escape($pattern)) {
            $missingPatterns += $pattern
        }
    }
    
    if ($missingPatterns.Count -eq 0) {
        Write-Host "   ✅ All critical patterns present in .gitignore`n" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Missing patterns in .gitignore:" -ForegroundColor Yellow
        $missingPatterns | ForEach-Object { Write-Host "      - $_" -ForegroundColor Yellow }
    }
} else {
    Write-Host "   ❌ .gitignore NOT FOUND!`n" -ForegroundColor Red
}

# 7. Check Git status
Write-Host "7. Checking Git repository status..." -ForegroundColor Cyan
$gitStatus = git status --porcelain 2>&1
if ($LASTEXITCODE -eq 0) {
    $uncommittedFiles = ($gitStatus | Measure-Object).Count
    if ($uncommittedFiles -gt 0) {
        Write-Host "   📝 You have $uncommittedFiles uncommitted changes" -ForegroundColor Yellow
        Write-Host "   💡 Run 'git status' to see details`n" -ForegroundColor Yellow
    } else {
        Write-Host "   ✅ Working directory is clean`n" -ForegroundColor Green
    }
} else {
    Write-Host "   ℹ️  Git status unavailable (might not be a git repo)`n" -ForegroundColor Gray
}

# 8. Project structure summary
Write-Host "8. Final Project Structure:" -ForegroundColor Cyan
$dirs = Get-ChildItem -Directory | Where-Object { $_.Name -notlike ".*" -and $_.Name -ne "venv" }
$dirs | ForEach-Object {
    $fileCount = (Get-ChildItem -Path $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "   📁 $($_.Name) - $fileCount files" -ForegroundColor White
}

# Final Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "CLEANUP SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✅ Removed $($stats.pycache_removed) __pycache__ directories" -ForegroundColor Green
Write-Host "✅ Removed $($stats.temp_files_removed) temporary files" -ForegroundColor Green
Write-Host "✅ Removed $($checkpoints.Count) Jupyter checkpoints" -ForegroundColor Green

if ($stats.duplicate_venv_found) {
    Write-Host "⚠️  Duplicate virtual environment detected" -ForegroundColor Yellow
}

if ($stats.large_files_found.Count -gt 0) {
    Write-Host "⚠️  $($stats.large_files_found.Count) large files detected" -ForegroundColor Yellow
}

Write-Host "`n📋 NEXT STEPS FOR SUBMISSION:" -ForegroundColor Yellow
Write-Host "   1. Review changes: git status" -ForegroundColor White
Write-Host "   2. Add files: git add ." -ForegroundColor White
Write-Host '   3. Commit: git commit -m "Final project cleanup for submission"' -ForegroundColor White
Write-Host "   4. Push to GitHub: git push origin main" -ForegroundColor White
Write-Host "   5. Verify on GitHub: Check repository online" -ForegroundColor White
Write-Host "`n✨ Project is ready for GitHub submission!" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Cyan
