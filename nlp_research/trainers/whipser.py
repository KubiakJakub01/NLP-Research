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
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        report_to=['tensorboard'],
        output_dir=hparams.output_dir,
        per_device_train_batch_size=hparams.batch_size,
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        learning_rate=hparams.learning_rate,
        warmup_steps=hparams.warmup_steps,
        max_steps=hparams.max_steps,
        gradient_checkpointing=hparams.gradient_checkpointing,
        fp16=hparams.use_fp16,
        evaluation_strategy=hparams.evaluation_strategy,
        per_device_eval_batch_size=hparams.eval_batch_size,
        predict_with_generate=hparams.predict_with_generate,
        generation_max_length=hparams.max_target_length,
        save_steps=hparams.save_steps,
        eval_steps=hparams.save_steps,
        logging_steps=hparams.logging_steps,
        push_to_hub=hparams.push_to_hub,
    )
    return Seq2SeqTrainer(
        args=training_args,
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        data_collator=collator,
    )
