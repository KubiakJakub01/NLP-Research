"""Implementation of GPT model."""
import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from ..hparams import GPTHparams


class GPT2LMHeadModel(GPT2Model):
    """GPT2 with language modeling head."""

    def __init__(self, config: GPTHparams):
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings."""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """Forward pass of GPT2LMHeadModel."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return BaseModelOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class GPT2LMHeadModelWithLatent(GPT2Model):
    """GPT2 with language modeling head and latent vector."""

    def __init__(self, config: GPTHparams):
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd + config.latent_size, config.vocab_size, bias=False)
        self.latent_size = config.latent_size
        self.latent = nn.Parameter(torch.zeros(1, 1, config.latent_size))

        self.init_weights()

    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings."""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        latent=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """Forward pass of GPT2LMHeadModel."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if latent is None:
            latent = self.latent

        latent = latent.expand(-1, -1, input_ids.shape[-1])

        inputs_embeds = self.wte(input_ids)
        inputs_embeds = torch.cat((inputs_embeds, latent), dim=-1)

        outputs = self.transformer(
            inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            latent=latent,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return BaseModelOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
