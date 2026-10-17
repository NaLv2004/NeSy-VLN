"""Direct SSH transport for the isolated predictive-rule experiment.

Credentials are read from the invoking process, never written to the workspace.
"""
import argparse
import os
import socket
from pathlib import Path

import paramiko


def connect():
    sock = socket.socket()
    sock.settimeout(20)
    if os.environ.get("STRIPS_SSH_LOCAL_IP"):
        sock.setsockopt(socket.IPPROTO_IP, 31,
                        socket.htonl(int(os.environ.get("STRIPS_SSH_INTERFACE", "10"))))
        try:
            sock.bind((os.environ["STRIPS_SSH_LOCAL_IP"], 0))
        except OSError as error:
            if getattr(error, 'winerror', None) != 10049:
                raise
            # Campus DHCP can change while the persistent dashboard is open.
            # Keep the explicit physical interface; let Windows choose its
            # current address instead of repeatedly binding the expired one.
            sock.bind(('0.0.0.0', 0))
    sock.connect(("10.129.2.34", 22))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("10.129.2.34", username="user", sock=sock,
                   password=os.environ["STRIPS_SSH_PASSWORD"],
                   look_for_keys=False, allow_agent=False, timeout=20)
    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script")
    parser.add_argument("--command")
    parser.add_argument("--upload", nargs=2, action="append", default=[])
    parser.add_argument("--download", nargs=2, action="append", default=[])
    args = parser.parse_args()
    client = connect()
    try:
        if args.upload or args.download:
            with client.open_sftp() as ftp:
                for source, destination in args.upload:
                    parent = Path(destination).as_posix().rsplit("/", 1)[0]
                    import shlex
                    _, out, err = client.exec_command("mkdir -p " + shlex.quote(parent))
                    if out.channel.recv_exit_status():
                        raise RuntimeError(err.read().decode())
                    ftp.put(source, destination)
                for source, destination in args.download:
                    Path(destination).parent.mkdir(parents=True, exist_ok=True)
                    ftp.get(source, destination)
        command = args.command
        if args.script:
            command = "python3 - <<'PREDICTIVE_REMOTE_SCRIPT'\n" + Path(args.script).read_text(encoding="utf-8") + "\nPREDICTIVE_REMOTE_SCRIPT"
        if command:
            _, out, err = client.exec_command(command, timeout=180)
            print(out.read().decode("utf-8", errors="replace"))
            error = err.read().decode("utf-8", errors="replace")
            if error:
                print(error)
            raise SystemExit(out.channel.recv_exit_status())
    finally:
        client.close()


if __name__ == "__main__":
    main()
