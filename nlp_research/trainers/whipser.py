from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


def get_whisper_trainer(hparams):
    model = WhisperForConditionalGeneration.from_pretrained(hparams.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(hparams.model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(hparams.model_name)

    training_args = Seq2SeqTrainingArguments(output_dir=hparams.output_dir)
    return Seq2SeqTrainer(
        args=training_args, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
    )
