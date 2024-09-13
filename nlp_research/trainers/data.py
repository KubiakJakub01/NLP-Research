from datasets import DatasetDict, load_dataset


def create_dataloader(hparams):
    dataset = DatasetDict()
    dataset['train'] = load_dataset(
        hparams.dataset_name, split='train', use_auth_token=hparams.use_auth_token
    )
    dataset['validation'] = load_dataset(
        hparams.dataset_name, split='validation', use_auth_token=hparams.use_auth_token
    )
    dataset = dataset.remove_columns(hparams.remove_columns)
