"""Local-only dashboard, persistent read-only SSH telemetry, automatic reconnect."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from metrics import prepare

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[2] / 'tools'))
from predictive_remote import connect

CACHE = BASE / 'cache.json'
LOCK = threading.RLock()
STATE = dict(train={}, development={}, development_transitions={}, attempts={}, trace_coverage={}, connected=False, received_at=None)
if CACHE.exists():
    try:
        STATE.update(json.loads(CACHE.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        pass
STATE.update(connected=False, error='正在连接远端。')
VERSION = 0
API_CACHE = {}


def ingest(packet):
    global VERSION
    with LOCK:
        STATE.update(connected=True, received_at=time.time(), error=None)
        if packet['kind'] == 'hello':
            STATE['info'] = packet
        elif packet['kind'] == 'telemetry':
            STATE['telemetry'] = packet['telemetry']
            STATE['pending_history_updates'] = packet['pending_history_updates']
            for kind in ('train', 'development'):
                for row in packet[kind]:
                    STATE[kind][str(int(row['update']))] = row
        elif packet['kind'] == 'attempts':
            for attempt in packet['attempts']:
                STATE['attempts'][attempt['key']] = attempt
            STATE['trace_coverage'][str(packet['update'])] = dict(expected=packet['expected'], found=packet['found'],
                audited=packet.get('audited_against_training_log', False))
            if len(STATE['attempts']) > 20000:
                keep = sorted(STATE['attempts'].values(), key=lambda a: (a['update'], a['rank'], a['index']))[-20000:]
                STATE['attempts'] = {a['key']: a for a in keep}
        elif packet['kind'] == 'development_transitions':
            STATE['development_transitions'][str(packet['update'])] = packet['transitions']
        elif packet['kind'] == 'factored':
            existing=STATE.get('factored',{})
            if existing.get('remote_path') and existing['remote_path'] != packet['remote_path']:
                STATE.setdefault('previous_factored_runs',{})[existing['remote_path']]=existing
                STATE['factored']=dict(train={},development={})
            current = STATE.setdefault('factored', dict(train={}, development={}))
            for kind in ('train', 'development'):
                for row in packet[kind]:
                    current[kind][str(row['update'])] = row
            current.update({k:v for k,v in packet.items() if k not in ('train','development','kind')})
        VERSION += 1
        API_CACHE.clear()


def receiver():
    last_save = 0
    while True:
        client = None
        try:
            if not os.environ.get('STRIPS_SSH_PASSWORD'):
                raise RuntimeError('SSH credential is missing')
            client = connect()
            client.get_transport().set_keepalive(20)
            remote = '/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/dashboard/remote_stream.py'
            _, stdout, stderr = client.exec_command('python3 -u ' + remote, timeout=45)
            for line in stdout:
                ingest(json.loads(line))
                if time.time() - last_save >= 30:
                    with LOCK:
                        snapshot = copy.deepcopy(STATE)
                    temp = CACHE.with_suffix('.tmp')
                    temp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
                    temp.replace(CACHE)
                    last_save = time.time()
            reason = stderr.read().decode(errors='replace')[-1500:]
            raise ConnectionError('Telemetry stream closed: ' + reason)
        except Exception as error:
            with LOCK:
                STATE.update(connected=False, error='远端连接暂不可用，正在自动重连。')
            print(time.strftime('%H:%M:%S'), type(error).__name__, str(error), flush=True)
        finally:
            if client:
                client.close()
        time.sleep(10)


def payload(k):
    with LOCK:
        key = (VERSION, k)
        if key in API_CACHE:
            result = copy.deepcopy(API_CACHE[key])
        else:
            result = prepare(STATE, k)
            result.update(info=STATE.get('info', {}), telemetry=STATE.get('telemetry', {}),
                          received_at=STATE.get('received_at'), pending_history_updates=STATE.get('pending_history_updates'))
            result['factored'] = copy.deepcopy(STATE.get('factored', {}))
            for kind in ('train', 'development'):
                result['factored'][kind] = sorted(result['factored'].get(kind, {}).values(), key=lambda r:r['update'])
            API_CACHE[key] = copy.deepcopy(result)
        result.update(connected=STATE['connected'], error=STATE.get('error'), local_time=time.time())
        result['stale'] = not result['connected'] or not result['received_at'] or time.time() - result['received_at'] > 25
        return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/api/status':
            try:
                k = min(20000, max(1, int(parse_qs(url.query).get('k', ['1024'])[0])))
            except ValueError:
                self.send_error(400, 'Invalid window'); return
            content = json.dumps(payload(k), ensure_ascii=False, allow_nan=False).encode('utf-8')
            mime = 'application/json; charset=utf-8'
        elif url.path in ('/', '/index.html', '/app.js', '/style.css'):
            file = 'index.html' if url.path in ('/', '/index.html') else url.path[1:]
            content = (BASE / file).read_bytes()
            mime = {'index.html': 'text/html', 'app.js': 'application/javascript', 'style.css': 'text/css'}[file] + '; charset=utf-8'
        elif url.path == '/favicon.ico':
            self.send_response(204); self.end_headers(); return
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8893)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=receiver, daemon=True, name='ppo-log-stream').start()
    (BASE / 'server.json').write_text(json.dumps(dict(pid=os.getpid(), port=args.port, started=time.time())), encoding='utf-8')
    print(f'PPO dashboard http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()
