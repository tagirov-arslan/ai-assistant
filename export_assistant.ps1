<#
    Упаковка ИИ-ассистента для переноса на другой компьютер.

    Создаёт zip со всем необходимым: исходники, модели (Whisper + Supertonic),
    скрипты установки и шаблон настроек. НЕ включает: виртуальные окружения,
    кэши, git, отладочные логи, бэкапы и личный assistant_settings.json
    (вместо него едет assistant_settings.example.json).

    Запуск из папки проекта:
        powershell -ExecutionPolicy Bypass -File export_assistant.ps1
    Параметры:
        -NoModels   собрать только код (без папок моделей ~1.2 ГБ)
#>

param(
    [switch]$NoModels
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$exportDir = Join-Path $projectDir "export"
$stageDir = Join-Path $exportDir "ai-assistant"
$zipPath = Join-Path $exportDir "ai-assistant-export-$stamp.zip"

$modelDirs = @("whisper-large-v3-turbo-ct2", "supertonic-3-model")

Write-Host "=== Экспорт ИИ-ассистента ===" -ForegroundColor Cyan
Write-Host "Проект: $projectDir"

# Чистая площадка для сборки
if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

# 1. Код и сопутствующие файлы — явный allowlist (без venv/git/кэшей/мусора).
$extraFiles = @("requirements.txt", ".env.example", "README.md",
                "assistant_settings.example.json", ".gitignore")
$codeFiles = Get-ChildItem -File -Path $projectDir | Where-Object {
    ($_.Extension -in @(".py", ".bat", ".ps1") -or $extraFiles -contains $_.Name) -and
    $_.Name -ne "assistant_settings.json" -and
    $_.Name -notlike "*.codex-backup-*" -and
    $_.Name -notlike "*.log"
}
foreach ($file in $codeFiles) {
    Copy-Item -LiteralPath $file.FullName -Destination $stageDir
}
Write-Host ("Скопировано файлов кода: {0}" -f $codeFiles.Count)

# 2. Модели (большие) — если не отключены.
if (-not $NoModels) {
    foreach ($model in $modelDirs) {
        $src = Join-Path $projectDir $model
        if (Test-Path $src) {
            Write-Host "Копирую модель '$model' ..."
            Copy-Item -LiteralPath $src -Destination $stageDir -Recurse
        } else {
            Write-Host "[ВНИМАНИЕ] Папка модели '$model' не найдена — докопируйте вручную." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "Режим -NoModels: папки моделей не включены."
}

# 3. Архивация.
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Write-Host "Архивирую ..."
Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath -CompressionLevel Optimal

Remove-Item -Recurse -Force $stageDir

$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Готово: $zipPath ($sizeMb МБ)" -ForegroundColor Green
Write-Host "На другом ПК: распакуйте, запустите setup.bat, затем run_app.bat."
if ($NoModels) {
    Write-Host "Внимание: собрано БЕЗ моделей — скопируйте папки whisper-large-v3-turbo-ct2 и supertonic-3-model отдельно." -ForegroundColor Yellow
}
