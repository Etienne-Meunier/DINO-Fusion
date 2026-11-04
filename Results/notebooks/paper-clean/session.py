#!/usr/bin/env python3
"""Persistent IPython kernel for the paper-clean scripts: `common` (data batch, generations) is loaded once, then each figure
script is re-run in that kernel in a few seconds.

    python session.py start              launch the kernel in the background and import common
    python session.py run fig5_aposteriori.py [...]
                                         %run each script in a fresh namespace (common is kept)
    python session.py exec "<code>"      execute code in the kernel, e.g. "importlib.reload(common)"
    python session.py stop               shut the kernel down

The kernel runs in this folder with the Agg backend; its connection file is .kernel.json next to this script. Edits to a
figure script are picked up by `run`; edits to common.py need `exec "importlib.reload(common)"` (the data is kept).
"""
import os
import subprocess
import sys
import time

from jupyter_client import BlockingKernelClient

HERE = os.path.dirname(os.path.abspath(__file__))
CONNECTION = f'{HERE}/.kernel.json'
LOG = f'{HERE}/.kernel.log'


def client():
    """BlockingKernelClient connected to the running kernel, channels started."""
    if not os.path.exists(CONNECTION):
        sys.exit('no kernel running, start it with: python session.py start')
    kc = BlockingKernelClient(connection_file=CONNECTION)
    kc.load_connection_file()
    kc.start_channels()
    return kc


def execute(code, timeout=None):
    """Execute `code` in the kernel, streaming its output; returns False on an error."""
    kc = client()
    ok = True

    def output(msg):
        nonlocal ok
        kind, content = msg['msg_type'], msg['content']
        if kind == 'stream':
            sys.stdout.write(content['text'])
        elif kind in ('execute_result', 'display_data'):
            text = content['data'].get('text/plain')
            if text:
                print(text)
        elif kind == 'error':
            ok = False
            print('\n'.join(content['traceback']))

    t0 = time.time()
    reply = kc.execute_interactive(code, output_hook=output, timeout=timeout)
    kc.stop_channels()
    print(f'[{time.time() - t0:.1f} s]')
    return ok and reply['content']['status'] == 'ok'


def start():
    """Launch a detached kernel in this folder, then import common in it."""
    if os.path.exists(CONNECTION):
        sys.exit(f'kernel already running ({CONNECTION}), stop it first')
    env = os.environ | {'MPLBACKEND': 'Agg'}
    with open(LOG, 'w') as log:
        subprocess.Popen([sys.executable, '-m', 'ipykernel_launcher', '-f', CONNECTION], cwd=HERE, env=env,
                         stdout=log, stderr=log, start_new_session=True)
    for _ in range(100):
        if os.path.exists(CONNECTION):
            break
        time.sleep(0.2)
    return execute('import importlib, common\nfrom common import *')


def run(scripts):
    """%run each script in the kernel, closing its figures afterwards."""
    return all(execute(f'%run {s}\nplt.close("all")') for s in scripts)


def stop():
    """Shut the kernel down and remove its connection file."""
    kc = client()
    kc.shutdown()
    # Without waiting for the reply, stop_channels can close the socket before the request is sent: the kernel lives on.
    kc.get_control_msg(timeout=30)
    kc.stop_channels()
    os.remove(CONNECTION)


if __name__ == '__main__':
    command, args = sys.argv[1], sys.argv[2:]
    if command == 'start':
        ok = start()
    elif command == 'run':
        ok = run(args)
    elif command == 'exec':
        ok = execute(' '.join(args))
    elif command == 'stop':
        ok = stop() is None
    else:
        sys.exit(__doc__)
    sys.exit(0 if ok else 1)
