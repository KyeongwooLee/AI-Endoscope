import argparse

from .common import lock, paths, read_json


def main():
    parser = argparse.ArgumentParser(description='Frozen Qwen SAE clustering; work starts only on explicit subcommands.')
    parser.add_argument('stage', choices=['prepare', 'extract', 'cluster', 'report'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--limit', type=int, help='Extraction only: positive multiple of shard_size (e.g. 32)')
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.limit is not None and (args.stage != 'extract' or args.limit <= 0 or args.limit % cfg['shard_size']):
        parser.error('--limit must be a positive multiple of shard_size, for extract only')
    _, out = paths(cfg)
    with lock(out):
        if args.stage == 'prepare':
            from .prepare import run
            run(cfg)
        elif args.stage == 'extract':
            from .extract import run
            run(cfg, args.limit)
        elif args.stage == 'cluster':
            from .cluster import run
            run(cfg)
        else:
            from .report import run
            run(cfg)


if __name__ == '__main__':
    main()
