# Nebula Journal

Personal website with:

- Apple-style landing page
- Login/register workspace (invite-code protected)
- Diary (create/edit/delete, per-user isolation)
- Simple web IDE (directory browser + Python/C++ execution)
- Interactive terminal run mode (send input during execution)
- Conda environment switching for Python/C++ run
- Admin console (user enable/disable, unlock, reset temp password)

## Local Run

```powershell
cd diary_site
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe app.py
```

Open: `http://127.0.0.1:8000`

## Remote Deploy (SSH Password)

```powershell
cd diary_site
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe deploy_remote.py `
  --host 100.71.92.98 `
  --user user `
  --password "your_ssh_password" `
  --sudo-password "your_sudo_password" `
  --remote-dir /home/user/diary_site `
  --app-port 18000 `
  --admin-user admin `
  --admin-password "your_site_password" `
  --registration-invite-code "your_invite_code"
```

Open after deploy: `http://SERVER_IP:18000`

## Environment Variables

See `.env.example` for basics. Optional runner tuning:

- `ENABLE_REGISTRATION` (default `1`)
- `REGISTRATION_INVITE_CODE` (empty means nobody can register)
- `USERNAME_MIN_LENGTH` / `USERNAME_MAX_LENGTH`
- `PASSWORD_MIN_LENGTH`
- `LOGIN_RATE_LIMIT` / `LOGIN_RATE_WINDOW_SECONDS`
- `REGISTER_RATE_LIMIT` / `REGISTER_RATE_WINDOW_SECONDS`
- `LOGIN_LOCK_THRESHOLD` / `LOGIN_LOCK_SECONDS`
- `RUNNER_RATE_LIMIT` / `RUNNER_RATE_WINDOW_SECONDS`
- `CODE_CONDA_BIN` (conda executable path, optional)
- `CONDA_DISCOVERY_TIMEOUT` / `CONDA_RUN_OVERHEAD`
- `IDE_WORKSPACE_ROOT`
- `IDE_MAX_FILE_CHARS` / `IDE_MAX_TREE_ENTRIES`
- `TERMINAL_MAX_BUFFER_CHARS` / `TERMINAL_MAX_OUTPUT_CHARS`
- `TERMINAL_IDLE_SECONDS` / `TERMINAL_MAX_SECONDS`
- `CODE_MAX_CHARS` (default `20000`)
- `CODE_MAX_STDIN_CHARS` (default `8000`)
- `CODE_RUN_TIMEOUT` (default `3.0`)
- `CPP_COMPILE_TIMEOUT` (default `8.0`)
- `CODE_MEMORY_MB` (default `256`)

## Service Commands

```bash
sudo systemctl status diary-site.service
sudo systemctl restart diary-site.service
sudo journalctl -u diary-site.service -f
```
