import torch
import torch.nn.functional as F


class DPOLoss(torch.nn.Module):
    """
    Direct Preference Optimization loss.
    """

    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        log_pi_theta_chosen: torch.Tensor,
        log_pi_theta_rejected: torch.Tensor,
        log_pi_ref_chosen: torch.Tensor,
        log_pi_ref_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the DPO loss.
        """

        # Log-ratio for chosen and rejected responses
        log_ratio_chosen = log_pi_theta_chosen - log_pi_ref_chosen
        log_ratio_rejected = log_pi_theta_rejected - log_pi_ref_rejected

        # DPO logits
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)

        # BCE loss
        loss = -F.logsigmoid(logits)

        return loss.mean()
