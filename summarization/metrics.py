from arguments import args
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# metric
class RankingLoss(nn.Module):

    def __init__(self, margin_lambda: float = 0.01) -> None:
        super(RankingLoss, self).__init__()

        self.margin_lambda = margin_lambda

    def forward(self, candidates_scores: torch.Tensor, summary_scores: torch.Tensor) -> torch.Tensor:

        batch_size, num_candidates = candidates_scores.size()

        # computes candidates vs summary loss
        summary_scores = summary_scores.unsqueeze(1).expand(batch_size, num_candidates)
        ranking_target = torch.ones_like(candidates_scores)
        loss = F.margin_ranking_loss(summary_scores, candidates_scores, target=ranking_target, margin=0.)

        # computes candidates ranking loss
        for i in range(1, num_candidates):
            ranking_target = torch.ones_like(candidates_scores[:, :-i])
            loss += F.margin_ranking_loss(candidates_scores[:, :-i], candidates_scores[:, i:],
                                          target=ranking_target, margin=i * self.margin_lambda)

        return loss
