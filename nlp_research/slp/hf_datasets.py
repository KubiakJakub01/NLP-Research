import argparse
import os
import subprocess
from functools import partial
from pathlib import Path

import gradio as gr
from datasets import Audio, load_dataset


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v=',
):
    status = False

    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" \
        --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()

    attempts = 0
    while True:
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def download(
    data_dir: Path,
    sampling_rate: int = 44100,
    limit: int | None = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.
    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        print(f'Limiting to {limit} examples')
        ds = ds.select(range(limit))

    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, _ = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(
        process, num_proc=num_proc, writer_batch_size=writer_batch_size, keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))


def get_example(idx, ds):
    ex = ds[idx]
    return ex['audio']['path'], ex['caption']


def main(
    data_dir: Path,
    sampling_rate: int = 44100,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    ds = download(data_dir, sampling_rate, limit, num_proc, writer_batch_size)

    gr.Interface(
        partial(get_example, ds=ds),
        inputs=gr.Slider(0, len(ds) - 1, value=0, step=1),
        outputs=['audio', 'textarea'],
        live=True,
    ).launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='data/music-caps')
    parser.add_argument('--sampling_rate', type=int, default=44100)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num_proc', type=int, default=1)
    parser.add_argument('--writer_batch_size', type=int, default=1000)
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        sampling_rate=args.sampling_rate,
        limit=args.limit,
        num_proc=args.num_proc,
        writer_batch_size=args.writer_batch_size,
    )
