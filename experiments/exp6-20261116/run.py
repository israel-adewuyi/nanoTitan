"""One fixed MoE architecture; only model dtype varies between precision runs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.sweep import dump_toml
from src.config import AppConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--master-port', type=int, default=29620)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    devices = args.gpus.split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or not all(x.isdigit() for x in devices):
        parser.error('The fixed EP2 configuration needs exactly two distinct GPU ordinals')
    raw = tomllib.loads(Path(__file__).with_name('base.toml').read_text())
    raw['model']['dtype'] = args.dtype
    raw['run_name'] = 'exp6-fixed-' + args.dtype
    raw['data']['dataset_path'] = str((REPO / raw['data']['dataset_path']).resolve())
    config = AppConfig.model_validate(raw)
    run_dir = args.output_dir.resolve() / config.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / 'resolved.toml'
    path.write_text(dump_toml(raw))
    prefix = ['uv', '--no-config', 'run', '--no-project', '--active', 'python']
    command = prefix + (['-m', 'src.train', str(path), '--dry-run'] if args.dry_run else [
        '-m', 'torch.distributed.run', '--nproc-per-node=2', f'--master-port={args.master_port}',
        '-m', 'src.train', str(path)])
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    record = dict(commit=sha, command=command, dtype=args.dtype, gpus=devices)
    (run_dir / 'launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    print(shlex.join(command), flush=True)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=args.gpus, CUDA_VISIBLE_DEVICES=args.gpus,
               PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / 'src'), os.environ.get('PYTHONPATH', '')]))
    raise SystemExit(subprocess.call(command, cwd=run_dir, env=env))


if __name__ == '__main__':
    main()
