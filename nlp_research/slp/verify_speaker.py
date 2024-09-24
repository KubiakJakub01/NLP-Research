import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Verify speaker')
    parser.add_argument('--input', type=str, help='Input file tsv file')
    parser.add_argument('--ref_col', type=str, help='Reference column')
    parser.add_argument('--compare_col', type=str, help='Column to compare')
    parser.add_argument('--simmilarity_col', type=str, help='Column to store simmilarity')
    return parser.parse_args()


def main():
    pass
