"""
A class for the reference policy model or trainable policy model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Policy(torch.nn.Module):
    """
    Causal LM policy that can act as: trainable policy or frozen reference policy
    """

    def __init__(
        self,
        model_path: str,
        max_length: int,
        trainable: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.trainable = trainable

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not trainable:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def log_prob(self, prompt: str, response: str) -> torch.Tensor:
        """
        given a state (prompt) and action (response), compute log-probability
        of response given prompt under the reference model
        """

        device = self.model.device

        # Tokenize prompt + response
        enc = self.tokenizer(
            prompt + response,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        # Forward pass, Disable grad if frozen reference
        with torch.set_grad_enabled(self.trainable):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits

        # Log-probs over vocabulary
        log_probs = torch.log_softmax(logits, dim=-1)

        # Tokenize response alone 
        response_ids = self.tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).input_ids.to(device)

        response_length = response_ids.size(1)

        # score response tokens only
        response_log_probs = log_probs[:, -response_length - 1 : -1, :]
        token_log_probs = response_log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum()
