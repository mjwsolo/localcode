import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog='tool')
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('greet'); g.add_argument('name')
    a = p.parse_args(argv)
    if a.cmd == 'greet':
        print(f'hello {a.name}')
        return 0
    return 1
