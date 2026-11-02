## Windows Setup (WSL2 — Recommended)

Run in PowerShell (Admin) if WSL is not installed:
```powershell
wsl --install
```

Inside the Ubuntu (WSL) terminal:
```bash
# 1. Install Micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

# 2. Create environment with Heyoka, Clang, Make, and Python plotting deps
micromamba create -n heyoka-env -c conda-forge heyoka clangxx make python numpy matplotlib -y

# 3. Build and run simulation
make run

# 4. Plot results
micromamba run -n heyoka-env python plotJetProfile.py
``'