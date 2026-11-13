import torch
import torch.nn as nn
import numpy as np


def build_isometric_q(source_dim: int, target_dim: int, seed: int) -> np.ndarray:
    """Return Q in R^(source_dim x target_dim) with orthonormal rows."""
    if target_dim < source_dim:
        raise ValueError(
            f"target_dim ({target_dim}) must be >= source_dim ({source_dim})"
        )
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((source_dim, target_dim))
    Q = np.zeros_like(A)
    for i in range(source_dim):
        v = A[i].copy()
        for j in range(i):
            v -= np.dot(Q[j], v) * Q[j]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise RuntimeError("Degenerate row during Gram-Schmidt construction")
        Q[i] = v / norm
    return Q.astype(np.float32)

class Forward_Block(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, p_val=0.0):
        super(Forward_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=p_val)
        )

    def forward(self, x):
        return self.block(x)

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out),
                         nn.BatchNorm1d(f_out),
                         nn.ReLU(inplace=True),
                         nn.Dropout(p=0.1))

class ResidualHadamardFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim * 2, bias=True),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(dim * 2, dim, bias=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
        self.w_img = nn.Parameter(torch.zeros(1))
        self.w_txt = nn.Parameter(torch.zeros(1))

    def forward(self, img, txt):
        interaction = img * txt
        fused = torch.cat([img, txt, interaction], dim=1)
        out   = self.proj(fused)
        out   = out + self.w_img * img + self.w_txt * txt
        out   = nn.functional.normalize(out, p=2, dim=1)
        return out, img, txt

class LinearWeightedAvg(nn.Module):
    def __init__(self, img_feat_dim, text_feat_dim):
        super(LinearWeightedAvg, self).__init__()
        self.weight1 = nn.Parameter(torch.rand(1))
        self.weight2 = nn.Parameter(torch.rand(1))

    def forward(self, img_feat, text_feat):
        return self.weight1 * img_feat + self.weight2 * text_feat, img_feat, text_feat

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.normalize(x)
        return x

class FOP(nn.Module):
    def __init__(self, args, img_feat_dim, text_feat_dim):
        super(FOP, self).__init__()
        self.use_msm  = getattr(args, 'use_msm',  False)
        self.msm_uplift = getattr(args, 'msm_uplift', False)

        self.image_branch = EmbedBranch(img_feat_dim,  args.dim_embed)
        self.text_branch  = EmbedBranch(text_feat_dim, args.dim_embed)

        if args.fusion == 'linear':
            self.fusion_layer = LinearWeightedAvg(args.dim_embed, args.dim_embed)
        elif args.fusion == 'hadamard':
            self.fusion_layer = ResidualHadamardFusion(args.dim_embed)

        if self.use_msm:
            if self.msm_uplift:
                canonical_dim = args.num_classes - 1
                msm_path = f'msms/prototypes{args.num_classes}_{canonical_dim}.npy'
                P = torch.tensor(np.load(msm_path), dtype=torch.float32)
                self.register_buffer("msm_canonical", P)
                self.msm_expand = nn.Linear(canonical_dim, args.dim_embed, bias=False)
            else:
                msm_path = f'msms/prototypes{args.num_classes}_{args.dim_embed}.npy'
                msm_matrix = torch.tensor(np.load(msm_path), dtype=torch.float32).T
                self.register_buffer("maxsep", msm_matrix)
        else:
            self.logits_layer = nn.Linear(args.dim_embed, args.num_classes)

        self.use_aux   = getattr(args, 'use_aux',    False)
        self.aux_weight = getattr(args, 'aux_weight', 0.0)
        if self.use_aux:
            self.img_aux = nn.Linear(args.dim_embed, args.num_classes)
            self.txt_aux = nn.Linear(args.dim_embed, args.num_classes)

        if args.cuda:
            self.cuda()

    def initialize_msm_projection(
        self, seed: int = 1, trainable: bool = False
    ) -> None:
        """Initialize canonical ETF projection with an orthogonal-row lift.

        Q has shape (C-1, D) and Q @ Q.T = I. The Linear weight is Q.T.
        Set trainable=True to optimize the projection from this initialization;
        otherwise it remains a fixed random orthogonal projection.
        """
        if not (self.use_msm and self.msm_uplift):
            return
        source_dim = int(self.msm_canonical.shape[1])
        target_dim = int(self.msm_expand.weight.shape[0])
        Q = build_isometric_q(source_dim, target_dim, seed)
        with torch.no_grad():
            self.msm_expand.weight.copy_(
                torch.from_numpy(Q.T).to(
                    device=self.msm_expand.weight.device,
                    dtype=self.msm_expand.weight.dtype,
                )
            )
        self.msm_expand.weight.requires_grad_(trainable)

    def initialize_msm_linear(self) -> None:
        """Trainable Xavier expansion; expanded prototypes are not ETF at init."""
        if not (self.use_msm and self.msm_uplift):
            return
        nn.init.xavier_uniform_(self.msm_expand.weight)
        self.msm_expand.weight.requires_grad_(True)

    def initialize_msm_uplift(self, seed: int = 1) -> None:
        """Backward-compatible alias for the fixed isometric uplift."""
        self.initialize_msm_projection(seed=seed, trainable=False)

    def _fused_logits(self, feats):
        if self.use_msm:
            if self.msm_uplift:
                P_expanded = nn.functional.normalize(
                    self.msm_expand(self.msm_canonical), p=2, dim=1)
                return torch.matmul(feats, P_expanded.T)
            else:
                return torch.matmul(feats, self.maxsep)
        else:
            return self.logits_layer(feats)

    def forward(self, images, texts):
        img_emb = self.image_branch(images)
        txt_emb = self.text_branch(texts)

        feats, img_out, txt_out = self.fusion_layer(img_emb, txt_emb)

        logits = self._fused_logits(feats)

        if self.use_aux:
            logits = logits + self.aux_weight * (
                self.img_aux(img_emb) + self.txt_aux(txt_emb)
            )

        return [feats, logits], img_emb, txt_emb

    def train_forward(self, images, texts, labels):
        comb, img_embeds, text_embeds = self(images, texts)
        return comb, img_embeds, text_embeds
