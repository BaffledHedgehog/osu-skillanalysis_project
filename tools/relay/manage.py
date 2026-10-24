"""Private reverse SSH tunnel: public relay -> isolated app on this PC.

Bootstrap reads the root password from an external file once. Runtime uses a
restricted SSH key. Configuration and keys live only in ignored .data/relay.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.data/relay'


def run_remote(client, command):
    _, output, error = client.exec_command(command, timeout=120)
    text = output.read().decode(errors='replace')
    detail = error.read().decode(errors='replace')
    if output.channel.recv_exit_status():
        raise RuntimeError('Relay command failed: ' + detail[-1000:])
    return text


def bootstrap(host, credentials_file):
    import paramiko
    STATE.mkdir(parents=True, exist_ok=True)
    lines = Path(credentials_file).read_text(encoding='utf-8-sig').splitlines()
    index = next((i for i, line in enumerate(lines) if f'root@{host}' in line), None)
    if index is None:
        raise ValueError('Requested root SSH account not found in the external credentials file.')
    password = lines[index + 1].partition(':')[2].strip()
    if password and password[0] == password[-1] and password[0] in '\"\'`':
        password = password[1:-1]
    if not password:
        raise ValueError('Password must follow the SSH account line.')
    key = STATE / 'id_rsa'
    if not key.exists():
        paramiko.RSAKey.generate(3072).write_private_key_file(str(key))
    if os.name == 'nt':
        owner = f'{os.environ["USERDOMAIN"]}\\{os.environ["USERNAME"]}'
        subprocess.run(['icacls', str(key), '/inheritance:r', '/grant:r', f'{owner}:(R)'], check=True, capture_output=True)
    else:
        key.chmod(0o600)
    public = paramiko.RSAKey.from_private_key_file(str(key))
    authorized = f'restrict,port-forwarding,permitlisten="127.0.0.1:18080",permitopen="127.0.0.1:1",command="/bin/false" {public.get_name()} {public.get_base64()} pulse-relay\n'
    known = STATE / 'known_hosts'
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if known.exists():
        client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username='root', password=password, allow_agent=False, look_for_keys=False, timeout=15)
        host_key = client.get_transport().get_remote_server_key()
        client.get_host_keys().add(host, host_key.get_name(), host_key)
        client.save_host_keys(str(known))
        # Only pulse-specific resources. Existing proxies and port 443 are untouched.
        run_remote(client, "command -v docker >/dev/null && (id pulse-tunnel >/dev/null 2>&1 || useradd -m -s /bin/sh pulse-tunnel) && install -d -m 700 -o pulse-tunnel -g pulse-tunnel /home/pulse-tunnel/.ssh && install -d -m 755 /opt/pulse-relay")
        with client.open_sftp() as sftp:
            with sftp.open('/home/pulse-tunnel/.ssh/authorized_keys', 'w') as remote:
                remote.write(authorized)
            sftp.chmod('/home/pulse-tunnel/.ssh/authorized_keys', 0o600)
            sftp.put(str(Path(__file__).with_name('nginx.conf')), '/opt/pulse-relay/nginx.conf')
        run_remote(client, 'chown pulse-tunnel:pulse-tunnel /home/pulse-tunnel/.ssh/authorized_keys')
        # Test config before replacing our own container.
        image = 'nginx:stable-alpine'
        run_remote(client, f'docker pull {image} >/dev/null')
        options = '--network host --user 101:101 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:uid=101,gid=101,mode=1770 --mount type=bind,src=/opt/pulse-relay/nginx.conf,dst=/etc/nginx/nginx.conf,readonly'
        run_remote(client, f'docker run --rm {options} {image} nginx -t')
        run_remote(client, 'if docker container inspect pulse-relay >/dev/null 2>&1; then docker rm -f pulse-relay >/dev/null; fi')
        run_remote(client, f'docker run -d --name pulse-relay --restart unless-stopped --memory 64m --cpus .3 --pids-limit 64 {options} {image} nginx -g "daemon off;"')
        (STATE / 'config.json').write_text(json.dumps({'host': host, 'user': 'pulse-tunnel', 'public_port': 8088, 'reverse_port': 18080, 'local_port': 8001}, indent=2), encoding='utf-8')
        print(f'Relay ready: http://{host}:8088 (HTTP demo; add a domain and TLS before private sessions).')
    finally:
        client.close()


def serve():
    config = json.loads((STATE / 'config.json').read_text(encoding='utf-8'))
    db = STATE / 'public.db'
    env = {**os.environ, 'PULSE_PUBLIC_MODE': '1', 'DATABASE_PATH': str(db)}
    app = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(config['local_port']), '--no-proxy-headers', '--limit-concurrency', '12'], cwd=ROOT, env=env)
    command = ['ssh', '-NT', '-i', str(STATE / 'id_rsa'), '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
               '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile={STATE / "known_hosts"}',
               '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
               '-R', f'127.0.0.1:{config["reverse_port"]}:127.0.0.1:{config["local_port"]}', f'{config["user"]}@{config["host"]}']
    tunnel = None
    try:
        while app.poll() is None:
            tunnel = subprocess.Popen(command)
            while app.poll() is None and tunnel.poll() is None:
                time.sleep(1)
            if tunnel.poll() is None:
                tunnel.terminate()
                tunnel.wait(timeout=10)
            if app.poll() is None:
                print('Tunnel disconnected; reconnecting in 5 seconds.', flush=True)
                time.sleep(5)
    finally:
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
        if app.poll() is None:
            app.terminate()
        app.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['bootstrap', 'serve'])
    parser.add_argument('--host')
    parser.add_argument('--credentials-file', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'bootstrap':
            if not args.host or not args.credentials_file:
                parser.error('bootstrap requires --host and --credentials-file')
            bootstrap(args.host, args.credentials_file)
        else:
            serve()
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
