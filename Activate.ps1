# Check if Python is installed
# Fonction pour vérifier la version de Python
function Get-PythonVersion {
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            return [version]$matches[0]
        } else {
            return $null
        }
    } catch {
        return $null
    }
}
function Uninstall-Python {
    $pythonInstallation = Get-WmiObject -Query "SELECT * FROM Win32_Product WHERE Name LIKE 'Python %'"
    
    if ($pythonInstallation) {
        Write-Host "Désinstallation de Python..." -ForegroundColor Yellow
        foreach ($installation in $pythonInstallation) {
            $installation.Uninstall() | Out-Null
            Write-Host "Python désinstallé : $($installation.Name)" -ForegroundColor Green
        }
    } else {
        Write-Host "Aucune installation de Python trouvée." -ForegroundColor Red
    }
}
function Install-Python {
    $pythonInstallerUrl = "https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe"
    $installerPath = "$env:TEMP\python-installer.exe"

    Write-Host "Téléchargement de l'installateur de Python..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $installerPath

    Write-Host "Installation de Python..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -NoNewWindow -Wait

    Remove-Item -Path $installerPath
    Write-Host "Python installé." -ForegroundColor Green

    # Vérifier que Python est correctement installé
    $pythonVersion = python --version
    if ($pythonVersion) {
        Write-Host "Python version : $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "L'installation de Python a échoué." -ForegroundColor Red
    }
}

# Vérifier si Python 3.12.4 est installé
$installedPythonVersion = Get-PythonVersion
$desiredPythonVersion = [version]"3.12.4"

if ($installedPythonVersion -eq $desiredPythonVersion) {
    Write-Host "Python $desiredPythonVersion est déjà installé." -ForegroundColor Green
} else {
    if ($installedPythonVersion) {
        Write-Host "Python $installedPythonVersion est installé. Désinstallation..." -ForegroundColor Yellow
        Uninstall-Python
    } else {
        Write-Host "Aucune version de Python n'est installée." -ForegroundColor Yellow
    }

    Write-Host "Installation de Python $desiredPythonVersion..." -ForegroundColor Yellow
    Install-Python

    # Vérifier que Python 3.12.4 est bien installé
    $installedPythonVersion = Get-PythonVersion
    if ($installedPythonVersion -eq $desiredPythonVersion) {
        Write-Host "Python $desiredPythonVersion a été installé avec succès." -ForegroundColor Green
    } else {
        Write-Host "L'installation de Python $desiredPythonVersion a échoué." -ForegroundColor Red
    }
}

# Télécharger le script d'installation de pip
Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
python get-pip.py
Remove-Item get-pip.py


# Remove the 'env' directory if it exists
if (Test-Path -Path "env") {
    Remove-Item -Recurse -Force "env"
}

# Create a virtual environment
python -m venv env

# Activate the virtual environment
& env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Navigate to the cloned repository
cd aibrigitte

# Install required packages
pip install -r requirements

# Run the training script
python ./train.py

# Start the GUI
python ./startGui.py
