param(
    [string]$MediaMtxDir = "C:\Users\USER\Downloads\mediamtx_v1.18.1_windows_amd64",
    [string]$MediaMtxExe = "mediamtx.exe",
    [string]$MediaMtxConfig = "mediamtx.yml",
    [string]$MonaInputUrl = "rtmp://127.0.0.1/live/drone",
    [string]$RawOutputUrl = "rtmp://127.0.0.1:1936/drone_raw",
    [string]$RvmOutputUrl = "rtmp://127.0.0.1:1936/rvm",
    [string]$QuestRvmWhepUrl = "http://192.168.1.7:8889/rvm/whep",
    [string]$QuestRawWhepUrl = "http://192.168.1.7:8889/drone_raw/whep",
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$Checkpoint = ".\rvm_mobilenetv3.pth",
    [string]$Variant = "mobilenetv3",
    [string]$Device = "auto",
    [int]$Width = 640,
    [int]$Height = 360,
    [double]$InputFps = 10,
    [double]$DownsampleRatio = 0.2,
    [double]$BitrateMbps = 1,
    [int]$ProcessEveryNthFrame = 1,
    [double]$ReconnectDelaySeconds = 2,
    [double]$FrameTimeoutSeconds = 8,
    [int]$InputQueueSize = 1
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  .\run_rvm_pipeline.ps1 -StartMediaMtx"
    Write-Host "  .\run_rvm_pipeline.ps1 -RawRepublisher"
    Write-Host "  .\run_rvm_pipeline.ps1 -WatchRaw"
    Write-Host "  .\run_rvm_pipeline.ps1 -RunRvm"
    Write-Host "  .\run_rvm_pipeline.ps1 -WatchRvm"
    Write-Host "  .\run_rvm_pipeline.ps1 -PrintUrls"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_rvm_pipeline.ps1 -StartMediaMtx"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_rvm_pipeline.ps1 -RawRepublisher"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_rvm_pipeline.ps1 -RunRvm"
    Write-Host ""
}

function Write-MediaMtxConfig {
    $configPath = Join-Path $MediaMtxDir $MediaMtxConfig
    @"
rtmpAddress: :1936
webrtcAdditionalHosts: [127.0.0.1, 192.168.1.7]
webrtcLocalTCPAddress: :8189

paths:
  all_others: {}
"@ | Set-Content -Path $configPath
    Write-Host "MediaMTX config written to $configPath"
}

function Start-MediaMtxServer {
    Write-MediaMtxConfig
    Push-Location $MediaMtxDir
    try {
        & ".\$MediaMtxExe"
    }
    finally {
        Pop-Location
    }
}

function Start-RawRepublisherLoop {
    while ($true) {
        ffmpeg -fflags +genpts+discardcorrupt -err_detect ignore_err -rtmp_live live `
            -use_wallclock_as_timestamps 1 `
            -i $MonaInputUrl `
            -map 0:v:0 -an `
            -c:v libx264 -preset ultrafast -tune zerolatency `
            -profile:v baseline -bf 0 `
            -g 20 -keyint_min 20 -sc_threshold 0 `
            -pix_fmt yuv420p `
            -f flv $RawOutputUrl

        Write-Host "raw republisher exited, restarting in 2 seconds..."
        Start-Sleep -Seconds 2
    }
}

function Watch-Raw {
    ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext $RawOutputUrl
}

function Start-RvmRelay {
    & $PythonExe .\rtmp_avatar_stream.py `
        --variant $Variant `
        --checkpoint $Checkpoint `
        --device $Device `
        --input-rtmp $RawOutputUrl `
        --output-rtmp $RvmOutputUrl `
        --mode black `
        --silhouette-color 0 0 0 `
        --background-color 255 255 255 `
        --input-resize $Width $Height `
        --input-fps $InputFps `
        --input-queue-size $InputQueueSize `
        --frame-timeout-seconds $FrameTimeoutSeconds `
        --downsample-ratio $DownsampleRatio `
        --bitrate-mbps $BitrateMbps `
        --process-every-nth-frame $ProcessEveryNthFrame `
        --reconnect-delay-seconds $ReconnectDelaySeconds
}

function Watch-Rvm {
    ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext $RvmOutputUrl
}

function Print-Urls {
    Write-Host "Quest raw WHEP: $QuestRawWhepUrl"
    Write-Host "Quest RVM WHEP: $QuestRvmWhepUrl"
    Write-Host "Raw RTMP:      $RawOutputUrl"
    Write-Host "RVM RTMP:      $RvmOutputUrl"
}

[bool]$StartMediaMtx = $false
[bool]$RawRepublisher = $false
[bool]$WatchRaw = $false
[bool]$RunRvm = $false
[bool]$WatchRvm = $false
[bool]$PrintUrls = $false

foreach ($arg in $args) {
    switch ($arg) {
        "-StartMediaMtx" { $StartMediaMtx = $true }
        "-RawRepublisher" { $RawRepublisher = $true }
        "-WatchRaw" { $WatchRaw = $true }
        "-RunRvm" { $RunRvm = $true }
        "-WatchRvm" { $WatchRvm = $true }
        "-PrintUrls" { $PrintUrls = $true }
        default { }
    }
}

if (-not ($StartMediaMtx -or $RawRepublisher -or $WatchRaw -or $RunRvm -or $WatchRvm -or $PrintUrls)) {
    Show-Usage
    exit 0
}

if ($StartMediaMtx) {
    Start-MediaMtxServer
}

if ($RawRepublisher) {
    Start-RawRepublisherLoop
}

if ($WatchRaw) {
    Watch-Raw
}

if ($RunRvm) {
    Start-RvmRelay
}

if ($WatchRvm) {
    Watch-Rvm
}

if ($PrintUrls) {
    Print-Urls
}
