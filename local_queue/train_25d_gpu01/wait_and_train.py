"""Wait for GPUs 0 and 1, then run the configured 2.5D trainer once."""

import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
QUEUE = Path(__file__).resolve().parent
ENV = Path('/mnt/nvme/lloyd/anaconda3/envs/edm')
COMMAND = [str(ENV / 'bin/accelerate'), 'launch', '--multi_gpu',
           '--num_processes', '2', '--gpu_ids', '0,1', 'train_accelerate_25d.py']
POLL_SECONDS = 30
REQUIRED_IDLE_CHECKS = 3


def log(message):
    print(f'{datetime.datetime.now().astimezone().isoformat(timespec="seconds")} {message}', flush=True)


def save_state(status, **details):
    state = {'status': status, 'watcher_pid': os.getpid(), 'command': COMMAND,
             'updated_at': datetime.datetime.now().astimezone().isoformat(), **details}
    temporary = QUEUE / 'state.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(QUEUE / 'state.json')


def query(arguments):
    result = subprocess.check_output(
        ['/usr/bin/nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
        text=True, stderr=subprocess.STDOUT, timeout=20)
    return [[value.strip() for value in row] for row in csv.reader(result.splitlines()) if row]


def check_idle():
    rows = query(['--query-gpu=index,uuid,memory.used,utilization.gpu'])
    selected = {int(row[0]): row for row in rows if int(row[0]) in (0, 1)}
    if set(selected) != {0, 1}:
        raise RuntimeError('Both GPUs 0 and 1 must be visible')
    processes = query(['--query-compute-apps=gpu_uuid,pid'])
    busy_uuids = {row[0] for row in processes}
    details = []
    idle = True
    for index in (0, 1):
        _, uuid, memory, utilization = selected[index]
        has_process = uuid in busy_uuids
        idle = idle and not has_process and float(memory) <= 512 and float(utilization) <= 5
        details.append(f'GPU {index}: {memory} MiB, {utilization}% utilization, compute_job={has_process}')
    return idle, '; '.join(details)


def main():
    with (QUEUE / 'watcher.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('Another watcher already holds this queue; exiting.')
            return 1
        log('Queued: ' + ' '.join(COMMAND))
        consecutive = 0
        while True:
            try:
                idle, details = check_idle()
                consecutive = consecutive + 1 if idle else 0
                log(f'{details}; idle checks {consecutive}/{REQUIRED_IDLE_CHECKS}')
                save_state('waiting', idle_checks=consecutive, gpu_status=details)
            except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
                consecutive = 0
                log(f'GPU check failed; continuing to wait: {error}')
                save_state('waiting', idle_checks=0, gpu_status=str(error))
            if consecutive >= REQUIRED_IDLE_CHECKS:
                break
            time.sleep(POLL_SECONDS)
        environment = os.environ.copy()
        environment['PATH'] = str(ENV / 'bin') + os.pathsep + environment.get('PATH', '')
        environment['CONDA_PREFIX'] = str(ENV)
        environment['CONDA_DEFAULT_ENV'] = 'edm'
        environment['PYTHONUNBUFFERED'] = '1'
        environment.pop('CUDA_VISIBLE_DEVICES', None)
        log('Both GPUs are idle. Starting training.')
        process = subprocess.Popen(COMMAND, cwd=ROOT, env=environment)
        save_state('running', training_pid=process.pid)
        result = process.wait()
        save_state('completed' if result == 0 else 'failed', training_pid=process.pid, exit_code=result)
        log(f'Training exited with status {result}. This queue will not restart it.')
        return result


if __name__ == '__main__':
    sys.exit(main())
