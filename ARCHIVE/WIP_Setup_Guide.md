# Setup Guide for COMPAS & Cosmic Integration

### Navigation

- [Section 1: Windows Subsystem for Linux (WSL)](#section-1-windows-subsystem-for-linux-wsl-wsl)
- [Section 2: Visual Studio Code (VS Code)](#section-2-visual-studio-code-vs-code)
- [Section 3: VS Code Extensions](#section-3-vs-code-extensions-ctrlshiftx)

## Section 1: Windows Subsystem for Linux (WSL)

### Windows Subsystem for Linux (WSL)

1. Open PowerShell in administrator mode by right-clicking and selecting "Run as administrator".
2. Install WSL: `wsl --install`
3. Restart machine.

### Ubuntu Distribution of Linux (Ubuntu)

1. Open PowerShell in administrator mode.
2. Install WSL with Ubuntu: `wsl --install`
3. Create new Linux user account and password.

## Section 2: Visual Studio Code (VS Code)

1. Install [Visual Studio Code](https://code.visualstudio.com/Download).
2. Select Windows User Installer.
    - Choose x64 for x86-64 architecture.
    - Choose Arm64 for aarch64 architecture.
3. During installation, verify that VS Code is added to PATH.

## Section 3: VS Code Extensions (Ctrl+Shift+X)

### Connect VS Code to WSL

1. Search and install "WSL" extension.
2. Open Command Palette (Ctrl+Shift+P or F1).
3. Search "WSL: Connect to WSL"
4. Press Enter.

### Install Additional Extensions

#### Extensions for working with Python environments:

- Python
    - Pylance
    - Python Debugger
    - Python Environments

#### Extensions for working with Jupyter Notebook environments:

- Jupyter
    - Jupyter Keymap
    - Jupyter Notebook Renderers
    - Jupyter Slide Show
    - Jupyter Cell Tags

#### Extensions for enabling secure shell (SSH) connections:

- Remote - SSH
    - Remote - SSH: Editing Configuration Files
    - Remote Explorer

---

Install Dependencies 
# Update all Ubuntu repository package lists: 
sudo apt-get update 
# Install C++ compiler, and the boost, gsl, and hdf5 library packages. 
sudo apt-get install g++ libboost-all-dev libgsl-dev libhdf5-dev 
Clone COMPAS Code Repository 
# Change to home directory: 
cd ~ 
# Clone the COMPAS code repository: 
git clone https://github.com/TeamCOMPAS/COMPAS 
Define COMPAS Environment Variable 
# Open bash file located in home directory (~/.bashrc) with Windows Notepad. 
# Append the following: 
export COMPAS_ROOT_DIR_=~/COMPAS 
# Reload bash shell session: 
source ~/.bashrc 
# (Optional) Check if environment variable was correctly defined;  
# Returns COMPAS root directory: 
echo $COMPAS_ROOT_DIR 
Build COMPAS 
# (If Applicable) Open Makefile located in COMPAS source code directory 
# (~/COMPAS/src) with Windows Notepad, and modify package directory paths. 
# 
# Most likely the only modification needed would be to the 
# HDF5LIBDIR directory path, e.g. 
# /usr/lib/x86_64-linux-gnu/hdf5/serial 
# /usr/lib/aarch64-linux-gnu/hdf5/serial 
# /usr/lib/<architecture>-linux-gnu/hdf5/serial 
# 
# (If Applicable) List the installed package directory paths: 
dpkg -L libboost-all-dev 
dpkg -L libgsl-dev 
dpkg -L libhdf5-dev 
# Install make command: 
sudo apt install make 
# Change to COMPAS source code directory: 
cd $COMPAS_ROOT_DIR/src 
# Clear preexisting builds of COMPAS: 
make clean 
# Build (Compile & Link) COMPAS: 
make 
# (Recommended) Build COMPAS using (4) simultaneous threads: 
make -j4 
# IMPORTANT: It is not recommended to build with more than 4 CPU threads 
# since the compilation has high RAM consumption.