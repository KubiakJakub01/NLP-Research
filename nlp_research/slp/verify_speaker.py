import argparse
from pathlib import Path

import nemo.collections.asr as nemo_asr
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Verify speaker')
    parser.add_argument('--input', type=Path, help='Input file tsv file')
    parser.add_argument('--ref_col', type=str, help='Reference column with audio path')
    parser.add_argument('--compare_col', type=str, help='Column to compare with reference')
    parser.add_argument('--verify_col', type=str, help='Column to store verification result')
    return parser.parse_args()


def main(input_tsv: Path, ref_col: str, compare_col: str, verify_col: str):
    df = pd.read_csv(input_tsv, sep='\t')

    # Load speaker verification model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        'nvidia/speakerverification_en_titanet_large'
    ).to(device)

    # Verify speaker
    df[verify_col] = df.apply(
        lambda x: speaker_verification_model.verify_speakers(x[ref_col], x[compare_col]), axis=1
    )

    # Save result
    df.to_csv(input_tsv, sep='\t', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(
        input_tsv=args.input,
        ref_col=args.ref_col,
        compare_col=args.compare_col,
        verify_col=args.verify_col,
    )
