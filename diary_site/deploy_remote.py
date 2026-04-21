import argparse
import posixpath
import secrets
import shlex
from pathlib import Path

import paramiko


LOCAL_ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "instance"}
EXCLUDE_FILES = {".env", "deploy_remote.py"}


def walk_local_files():
    for path in LOCAL_ROOT.rglob("*"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.is_file() and path.name not in EXCLUDE_FILES:
            rel = path.relative_to(LOCAL_ROOT).as_posix()
            yield path, rel


def run(
    ssh: paramiko.SSHClient,
    command: str,
    *,
    use_sudo: bool = False,
    sudo_password: str = "",
):
    remote_command = command
    if use_sudo:
        remote_command = f"sudo -S -p '' bash -lc {shlex.quote(command)}"

    stdin, stdout, stderr = ssh.exec_command(remote_command)
    if use_sudo:
        stdin.write(sudo_password + "\n")
        stdin.flush()
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if exit_code != 0:
        raise RuntimeError(f"Remote command failed ({exit_code}): {command}\n{err or out}")
    return out


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str):
    chunks = remote_dir.strip("/").split("/")
    current = ""
    for chunk in chunks:
        current = f"{current}/{chunk}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def deploy(args):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=20,
        look_for_keys=False,
    )
    try:
        print("Connected. Installing base dependencies...")
        need_sudo = args.user != "root"
        sudo_password = args.sudo_password or args.password
        install_cmd = (
            "if command -v python3 >/dev/null 2>&1 && python3 -m venv --help >/dev/null 2>&1; then "
            "echo 'python3 and venv already available'; "
            "elif command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y python3 python3-venv python3-pip; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "dnf install -y python3 python3-pip; "
            "elif command -v yum >/dev/null 2>&1; then "
            "yum install -y python3 python3-pip; "
            "else echo 'No supported package manager found'; exit 1; fi"
        )
        run(ssh, install_cmd, use_sudo=need_sudo, sudo_password=sudo_password)

        sftp = ssh.open_sftp()
        try:
            ensure_remote_dir(sftp, args.remote_dir)
            for local_path, relative_path in walk_local_files():
                remote_path = posixpath.join(args.remote_dir, relative_path)
                remote_parent = posixpath.dirname(remote_path)
                ensure_remote_dir(sftp, remote_parent)
                sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()

        flask_secret = args.flask_secret or secrets.token_hex(32)
        env_content = (
            f"FLASK_SECRET_KEY={flask_secret}\n"
            f"ADMIN_USERNAME={args.admin_user}\n"
            f"ADMIN_PASSWORD={args.admin_password}\n"
            f"ENABLE_REGISTRATION={'0' if args.disable_registration else '1'}\n"
            f"REGISTRATION_INVITE_CODE={args.registration_invite_code}\n"
        )
        run(
            ssh,
            "cat > {remote}/.env <<'EOF'\n{env}\nEOF".format(
                remote=args.remote_dir,
                env=env_content.strip(),
            ),
        )

        print("Creating virtualenv and installing app dependencies...")
        run(
            ssh,
            (
                f"cd {args.remote_dir} && "
                "python3 -m venv .venv && "
                ".venv/bin/pip install --upgrade pip && "
                ".venv/bin/pip install -r requirements.txt"
            ),
        )

        service_content = f"""[Unit]
Description=Diary Website Service
After=network.target

[Service]
Type=simple
WorkingDirectory={args.remote_dir}
EnvironmentFile={args.remote_dir}/.env
ExecStart={args.remote_dir}/.venv/bin/gunicorn -w 1 -b 0.0.0.0:{args.app_port} app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""
        run(
            ssh,
            "cat > /etc/systemd/system/{name}.service <<'EOF'\n{content}\nEOF".format(
                name=args.service_name,
                content=service_content.strip(),
            ),
            use_sudo=need_sudo,
            sudo_password=sudo_password,
        )
        run(
            ssh,
            f"systemctl daemon-reload && systemctl enable --now {args.service_name}.service",
            use_sudo=need_sudo,
            sudo_password=sudo_password,
        )
        run(
            ssh,
            f"systemctl restart {args.service_name}.service",
            use_sudo=need_sudo,
            sudo_password=sudo_password,
        )
        status = run(
            ssh,
            f"systemctl --no-pager --full status {args.service_name}.service | head -n 12",
            use_sudo=need_sudo,
            sudo_password=sudo_password,
        )
        print(status)
        print(f"\nDeploy finished: http://{args.host}:{args.app_port}")
    finally:
        ssh.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy diary website to a Linux server through SSH.")
    parser.add_argument("--host", required=True, help="Server IP or hostname")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--user", default="root", help="SSH username")
    parser.add_argument("--password", required=True, help="SSH password")
    parser.add_argument("--remote-dir", default="/opt/diary_site", help="Remote deployment directory")
    parser.add_argument("--service-name", default="diary-site", help="Systemd service name")
    parser.add_argument("--app-port", type=int, default=18000, help="Port exposed by gunicorn")
    parser.add_argument("--sudo-password", default="", help="Sudo password when SSH user is not root")
    parser.add_argument("--admin-user", default="admin", help="Diary login username")
    parser.add_argument("--admin-password", required=True, help="Diary login password")
    parser.add_argument("--flask-secret", default="", help="Flask secret key; generated when omitted")
    parser.add_argument("--registration-invite-code", default="", help="Invite code required for registration")
    parser.add_argument("--disable-registration", action="store_true", help="Disable public registration")
    return parser.parse_args()


if __name__ == "__main__":
    deploy(parse_args())
