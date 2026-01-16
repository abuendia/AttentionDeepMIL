import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class FeatureAttention(nn.Module):
    """Attention MIL model for vector instance features (e.g., k-mer encodings).

    Expects input X shaped either:
      - (1, K, D) from a DataLoader with batch_size=1, or
      - (K, D)
    where K is number of instances (sequences) in the bag and D is feature dim.
    """

    def __init__(self, input_dim: int, M: int = 500, L: int = 128, attention_branches: int = 1):
        super(FeatureAttention, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, self.M),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x, lengths=None):
        """Forward pass.

        Args:
            x: (K, D) or (1, K, D) or (B, K, D)
            lengths: optional 1D tensor/list of length B with the true (unpadded) K per bag.
                     If provided, padded instances are masked out in attention softmax.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 1xKxD
        elif x.dim() != 3:
            raise ValueError(f"Expected x with 2 or 3 dims, got {x.dim()} (shape={tuple(x.shape)})")

        B, K, _ = x.shape

        H = self.feature_extractor(x)  # BxKxM

        attn_logits = self.attention(H)  # BxKxATTENTION_BRANCHES
        A = attn_logits.permute(0, 2, 1)  # BxATTENTION_BRANCHESxK

        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=A.device)
            lengths = lengths.to(device=A.device)
            # mask: BxK
            mask = torch.arange(K, device=A.device).unsqueeze(0) < lengths.view(-1, 1)
            A = A.masked_fill(~mask.unsqueeze(1), float('-inf'))

        A = F.softmax(A, dim=2)  # softmax over K

        Z = torch.bmm(A, H)  # BxATTENTION_BRANCHESxM
        Z = Z.reshape(B, self.M * self.ATTENTION_BRANCHES)  # Bx(ATTENTION_BRANCHES*M)

        Y_prob = self.classifier(Z)  # Bx1
        Y_hat = torch.ge(Y_prob, 0.0).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

        return neg_log_likelihood, A


class FeatureGatedAttention(nn.Module):
    """Gated-attention MIL model for vector instance features (e.g., k-mer encodings)."""

    def __init__(self, input_dim: int, M: int = 500, L: int = 128, attention_branches: int = 1):
        super(FeatureGatedAttention, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, self.M),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid(),
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f"Expected x with 2 or 3 dims, got {x.dim()} (shape={tuple(x.shape)})")

        B, K, _ = x.shape

        H = self.feature_extractor(x)  # BxKxM

        A_V = self.attention_V(H)  # BxKxL
        A_U = self.attention_U(H)  # BxKxL
        attn_logits = self.attention_w(A_V * A_U)  # BxKxATTENTION_BRANCHES
        A = attn_logits.permute(0, 2, 1)  # BxATTENTION_BRANCHESxK

        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=A.device)
            lengths = lengths.to(device=A.device)
            mask = torch.arange(K, device=A.device).unsqueeze(0) < lengths.view(-1, 1)
            A = A.masked_fill(~mask.unsqueeze(1), float('-inf'))

        A = F.softmax(A, dim=2)

        Z = torch.bmm(A, H)  # BxATTENTION_BRANCHESxM
        Z = Z.reshape(B, self.M * self.ATTENTION_BRANCHES)

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.0).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

        return neg_log_likelihood, A
