#!/usr/bin/env python3
"""Start the configured local mlx-serve service, or print its exact argv.

The existing listener must be stopped before starting. This command never kills
another process and does not launch an additional model on an occupied port.
"""
import argparse
import json
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]


def command(config):
    binary = (ROOT / config['binary']).resolve()
    model = (ROOT / config['model']).resolve()
    text = json.loads((model / 'config.json').read_text())
    native = text.get('text_config', text)['max_position_embeddings']
    if not 1 <= config['context_tokens'] <= native:
        raise ValueError('Context must fit the model native position limit')
    args = [str(binary), '--model', str(model), '--serve',
            '--host', config['host'], '--port', str(config['port'])]
    for key, flag in (
        ('context_tokens', '--ctx-size'), ('prefill_chunk', '--prefill-chunk'),
        ('max_concurrent', '--max-concurrent'), ('prefix_cache_entries', '--prefix-cache-entries'),
        ('prefix_cache_memory', '--prefix-cache-mem'), ('prefix_cache_disk', '--prefix-cache-disk'),
        ('ssm_checkpoint_stride', '--ssm-checkpoint-stride'), ('ssm_checkpoint_max', '--ssm-checkpoint-max'),
        ('timeout_seconds', '--timeout'),
    ):
        args.extend([flag, str(config[key])])
    return args + ['--no-mtp', '--no-drafter', '--no-pld', '--metrics']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/mlx-serve-business.json')
    parser.add_argument('--print-argv', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    argv = command(config)
    if args.print_argv:
        print(json.dumps(argv))
        return
    if not os.access(argv[0], os.X_OK):
        parser.error('Configured mlx-serve binary is not built or executable')
    try:
        with socket.create_connection((config['host'], config['port']), timeout=2):
            parser.error('Service port is occupied; retain or stop the existing service first')
    except ConnectionRefusedError:
        pass
    os.execv(argv[0], argv)


if __name__ == '__main__':
    main()
