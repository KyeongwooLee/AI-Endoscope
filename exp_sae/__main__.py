import argparse
from pathlib import Path

from .common import environment, load_config


def main():
    parser = argparse.ArgumentParser(description='Repeated GSM8K + reviewed taxonomy + post-generation SAE experiment')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    commands = parser.add_subparsers(dest='stage', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--registry-only', action='store_true')
    generate = commands.add_parser('generate')
    generate.add_argument('--worker', type=int, required=True)
    generate.add_argument('--limit', type=int, help='number of problems for this worker')
    discover = commands.add_parser('discover-taxonomy')
    discover.add_argument('action', choices=['build', 'synthesize', 'submit', 'collect', 'freeze'])
    discover.add_argument('--key')
    discover.add_argument('--draft', help='human-reviewed codebook JSON to freeze')
    labels = commands.add_parser('label')
    labels.add_argument('action', choices=['build', 'submit', 'collect', 'materialize', 'review-pilot'])
    labels.add_argument('--scope', choices=['discovery', 'pilot', 'confirmation'], default='pilot')
    labels.add_argument('--problems', type=int)
    labels.add_argument('--key')
    labels.add_argument('--review')
    extract = commands.add_parser('extract')
    extract.add_argument('--limit', type=int, help='first N global problem IDs (pilot: 2)')
    extract.add_argument('--sae', help='registry ID; default: all metadata-verified entries')
    commands.add_parser('analyze')
    commands.add_parser('report')
    args = parser.parse_args()
    cfg = load_config(args.config)
    environment(cfg)
    if args.stage == 'prepare':
        from . import prepare, sae
        sae.registry(cfg)
        if not args.registry_only:
            prepare.run(cfg)
    elif args.stage == 'generate':
        from .generate import run
        run(cfg, args.worker, args.limit)
    elif args.stage in ('discover-taxonomy', 'label'):
        from . import batch, label, taxonomy
        if args.action in ('submit', 'collect'):
            if not args.key:
                parser.error('--key is required')
            if args.action == 'submit':
                print(batch.submit(cfg, args.key))
            else:
                result = batch.collect(cfg, args.key)
                if result is not None:
                    if args.stage == 'label':
                        label.materialize(cfg)
                    else:
                        _, book = batch.ledger(cfg)
                        if book['batches'][args.key]['kind'] == 'taxonomy-synthesis':
                            print(taxonomy.export_draft(cfg, args.key))
        elif args.stage == 'discover-taxonomy':
            if args.action == 'freeze':
                if not args.draft:
                    parser.error('--draft is required')
                taxonomy.freeze(cfg, args.draft)
            else:
                print(taxonomy.build(cfg, args.action == 'synthesize'))
        elif args.action == 'build':
            print(label.build(cfg, args.scope, args.problems))
        elif args.action == 'materialize':
            print(label.materialize(cfg))
        else:
            print(label.review_pilot(cfg, args.review))
    elif args.stage == 'extract':
        from .extract import run
        run(cfg, args.limit, args.sae)
    else:
        from importlib import import_module
        import_module('.' + args.stage, __package__).run(cfg)


if __name__ == '__main__':
    main()
