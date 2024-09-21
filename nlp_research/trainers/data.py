from datasets import Audio, DatasetDict, load_dataset


def create_audio_dataset(hparams):
    dataset = DatasetDict()
    dataset['train'] = load_dataset(
        hparams.dataset_name, hparams.subset, split='train', use_auth_token=hparams.use_auth_token
    )
    dataset['validation'] = load_dataset(
        hparams.dataset_name,
        hparams.subset,
        split='validation',
        use_auth_token=hparams.use_auth_token,
    )
    dataset = dataset.remove_columns(hparams.remove_columns)
    dataset = dataset.cast_column('audio', Audio(hparams.sample_rate))

    return dataset
