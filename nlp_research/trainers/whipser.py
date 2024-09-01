from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

from .collators import DataCollatorSpeechSeq2SeqWithPadding


def get_whisper_trainer(hparams):
    model = WhisperForConditionalGeneration.from_pretrained(hparams.model_name)
    processor = AutoProcessor.from_pretrained(hparams.processor_name)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=processor.tokenizer.pad_token_id
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir='./whisper-small-hi',  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy='steps',
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        push_to_hub=True,
    )
    return Seq2SeqTrainer(
        args=training_args,
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        data_collator=collator,
    )
