import evaluate
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

from .collators import DataCollatorSpeechSeq2SeqWithPadding


class WhisperTrainer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = WhisperForConditionalGeneration.from_pretrained(self.hparams.model_name)
        self.processor = AutoProcessor.from_pretrained(self.hparams.processor_name)
        self.collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor, decoder_start_token_id=self.processor.tokenizer.pad_token_id
        )
        self.metric = evaluate.load('wer')

        training_args = Seq2SeqTrainingArguments(
            load_best_model_at_end=True,
            metric_for_best_model='wer',
            greater_is_better=False,
            report_to=['tensorboard'],
            output_dir=self.hparams.output_dir,
            per_device_train_batch_size=self.hparams.batch_size,
            gradient_accumulation_steps=self.hparams.gradient_accumulation_steps,
            learning_rate=self.hparams.learning_rate,
            warmup_steps=self.hparams.warmup_steps,
            max_steps=self.hparams.max_steps,
            gradient_checkpointing=self.hparams.gradient_checkpointing,
            fp16=self.hparams.use_fp16,
            evaluation_strategy=self.hparams.evaluation_strategy,
            per_device_eval_batch_size=self.hparams.eval_batch_size,
            predict_with_generate=self.hparams.predict_with_generate,
            generation_max_length=self.hparams.max_target_length,
            save_steps=self.hparams.save_steps,
            eval_steps=self.hparams.save_steps,
            logging_steps=self.hparams.logging_steps,
            push_to_hub=self.hparams.push_to_hub,
        )
        self._trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            compute_metrics=self.compute_metrics,
            data_collator=self.collator,
        )

    @property
    def tokenizer(self):
        return self.processor.feature

    @property
    def feature_extractor(self):
        return self.processor.feature_extractor

    def train(self):
        self._trainer.train(resume_from_checkpoint=self.hparams.resume_from_checkpoint)

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {'wer': wer}
