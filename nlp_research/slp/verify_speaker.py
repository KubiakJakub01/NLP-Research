import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Verify speaker')
    parser.add_argument('--input', type=str, help='Input file')
    parser.add_argument('--output', type=str, help='Output file')
    return parser.parse_args()
