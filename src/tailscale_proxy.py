"""Optional Colab userspace exit-node proxy; no system routes or Drive credentials."""
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import subprocess
import tarfile
import time
import urllib.request

VERSION = '1.104.4'
HOME = Path('/content/sidewalk-tailscale')
PROXY = 'http://127.0.0.1:10554'
_daemon = None


def install():
    if platform.system() != 'Linux':
        raise RuntimeError('Run the Tailscale cells in Colab/Linux.')
    arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine())
    if not arch:
        raise RuntimeError('Unsupported CPU architecture')
    HOME.mkdir(parents=True, exist_ok=True, mode=0o700)
    HOME.chmod(0o700)
    folder = HOME / f'tailscale_{VERSION}_{arch}'
    if not (folder / 'tailscale').exists():
        url = f'https://pkgs.tailscale.com/stable/tailscale_{VERSION}_{arch}.tgz'
        with urllib.request.urlopen(url + '.sha256', timeout=60) as response:
            expected = response.read().decode().split()[0]
        archive = HOME / 'tailscale.tgz'
        with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Tailscale package checksum mismatch')
        with tarfile.open(archive) as source:
            # Extract only the two expected regular files, never arbitrary paths or links.
            for name in ['tailscale', 'tailscaled']:
                member = source.getmember(f'{folder.name}/{name}')
                if not member.isfile():
                    raise RuntimeError('Unexpected binary archive entry')
                folder.mkdir(exist_ok=True)
                (folder / name).write_bytes(source.extractfile(member).read())
                (folder / name).chmod(0o700)
        archive.unlink()
    return folder


def _cli(*args, timeout=30):
    folder = install()
    return subprocess.run([str(folder / 'tailscale'), '--socket=' + str(HOME / 'tailscaled.sock'), *args],
                          capture_output=True, text=True, timeout=timeout)


def status():
    result = _cli('status', '--json')
    try:
        return json.loads(result.stdout)
    except ValueError:
        return {'BackendState': 'Unavailable'}


def start_login():
    global _daemon
    folder = install()
    # No auth key needed: display a browser login URL. State stays only in RAM.
    if status().get('BackendState') == 'Unavailable':
        with (HOME / 'daemon.log').open('a') as log:
            _daemon = subprocess.Popen([
                str(folder / 'tailscaled'), '--tun=userspace-networking', '--state=mem:',
                '--socket=' + str(HOME / 'tailscaled.sock'),
                '--outbound-http-proxy-listen=127.0.0.1:10554',
            ], stdout=log, stderr=subprocess.STDOUT)
        for _ in range(40):
            if _daemon.poll() is not None:
                raise RuntimeError('Tailscale daemon exited; inspect its local daemon.log')
            if status().get('BackendState') != 'Unavailable':
                break
            time.sleep(0.25)
        else:
            raise RuntimeError('Tailscale daemon did not start')
    current = status()
    if current.get('BackendState') == 'Running':
        print('Tailscale login is already active. Run the exit-node cell.')
        return True
    result = _cli('up', '--accept-dns=false', '--hostname=colab-sidewalk', '--timeout=15s', timeout=25)
    current = status()
    url = current.get('AuthURL') or next(iter(re.findall(r'https://login\.tailscale\.com/\S+', result.stdout + result.stderr)), '')
    if url:
        print('Open this link, sign in to your tailnet, then run the next cell:\n' + url)
    elif current.get('BackendState') != 'Running':
        raise RuntimeError('Tailscale login could not start; inspect local daemon.log')
    return current.get('BackendState') == 'Running'


def exit_nodes():
    current = status()
    if current.get('BackendState') != 'Running':
        raise RuntimeError('Complete the browser login first, then rerun this cell.')
    return [{'name': p.get('HostName'), 'dns_name': p.get('DNSName'),
             'ip': next(iter(p.get('TailscaleIPs', [])), ''), 'online': p.get('Online', False)}
            for p in (current.get('Peer') or {}).values() if p.get('ExitNodeOption')]


def select_exit_node(node):
    os.environ.pop('AIHUB_PROXY_URL', None)
    candidates = exit_nodes()
    matches = [p for p in candidates if node in (p['name'], p['dns_name'], p['dns_name'].rstrip('.'), p['ip'])]
    if len(matches) != 1 or not matches[0]['online']:
        raise ValueError('Choose one online, approved exit node from the displayed list.')
    result = _cli('set', '--exit-node=' + matches[0]['ip'], '--exit-node-allow-lan-access=false', '--accept-dns=false')
    if result.returncode:
        raise RuntimeError('Cannot enable exit node; check tailnet access policy and node approval.')
    current = status()
    selected = current.get('ExitNodeStatus') or {}
    if not selected.get('Online'):
        raise RuntimeError('Selected exit node is not online; AI Hub proxy remains disabled.')
    # Probe explicitly through the proxy. Do not change global HTTPS_PROXY or system routes.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY}))
    with opener.open('https://api.ipify.org?format=json', timeout=30) as response:
        public_ip = json.loads(response.read())['ip']
    os.environ['AIHUB_PROXY_URL'] = PROXY
    print('Exit node connected:', matches[0]['name'])
    print('Proxy egress public IP:', public_ip)
    print('AI Hub requests will use this proxy. Confirm this is your domestic exit-node IP.')
    return public_ip


def disconnect():
    os.environ.pop('AIHUB_PROXY_URL', None)
    result = _cli('down')
    if result.returncode:
        raise RuntimeError('Could not disconnect Tailscale')
    print('AI Hub proxy disabled; Tailscale disconnected.')
