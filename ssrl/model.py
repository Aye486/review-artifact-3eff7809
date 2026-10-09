import math

import torch
from torch import nn
from torch.nn import functional as F


PROMPT_GROUPS = (
    (
        "a normal industrial product without any visible defects",
        "a defect-free industrial component",
        "a product with a clean and intact surface",
    ),
    (
        "a damaged industrial component",
        "an industrial product with cracks scratches holes or contamination",
    ),
    (
        "a consistent 3D geometric structure without deformation",
        "a regular texture pattern and an intact geometric structure",
    ),
)


class SemanticAnchors(nn.Module):
    """Encode fixed prompts once; only the attention projection is trainable."""

    def __init__(self, dim, heads, clip_model="ViT-B/32", text_embeddings=None):
        super().__init__()
        if text_embeddings is None:
            import clip

            encoder, _ = clip.load(clip_model, device="cpu", jit=False)
            encoder.eval().requires_grad_(False)
            with torch.no_grad():
                groups = [
                    encoder.encode_text(clip.tokenize(list(group))).float()
                    for group in PROMPT_GROUPS
                ]
                text_embeddings = torch.stack(
                    [F.normalize(group, dim=-1).mean(0) for group in groups]
                )
            del encoder
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 3:
            raise ValueError(
                "text_embeddings must contain three CLIP prompt-group vectors"
            )
        self.register_buffer(
            "text_embeddings", F.normalize(text_embeddings.float(), dim=-1)
        )
        self.projection = nn.Linear(text_embeddings.shape[-1], dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch_size):
        z = self.projection(self.text_embeddings).unsqueeze(1)
        z = self.norm(z + self.attention(z, z, z, need_weights=False)[0])
        return z.expand(-1, batch_size, -1)


class SelectiveScan2D(nn.Module):
    """Four directional selective SSM with optional fused mamba-ssm kernel.

    The reference backend uses the same input-dependent delta/B/C recurrence;
    it is intended for CPU checks, not large-scale training.
    """

    def __init__(self, dim, state_dim=16, backend="auto"):
        super().__init__()
        if backend not in ("auto", "reference", "mamba"):
            raise ValueError("scan backend must be auto, reference or mamba")
        self.backend = backend
        self.state_dim = state_dim
        self.rank = max(1, math.ceil(dim / 16))
        self.projections = nn.ModuleList(
            [nn.Linear(dim, self.rank + 2 * state_dim, bias=False) for _ in range(4)]
        )
        self.deltas = nn.ModuleList([nn.Linear(self.rank, dim) for _ in range(4)])
        self.a_log = nn.Parameter(
            torch.arange(1, state_dim + 1).float().log().repeat(4, dim, 1)
        )
        self.skip = nn.Parameter(torch.ones(4, dim))
        for delta in self.deltas:
            nn.init.constant_(delta.bias, -3.0)
        self.norm = nn.LayerNorm(dim)

    def scan(self, sequence, direction):
        params = self.projections[direction](sequence)
        dt, b, c = torch.split(params, [self.rank, self.state_dim, self.state_dim], -1)
        dt = F.softplus(self.deltas[direction](dt)).float()
        a = -self.a_log[direction].float().exp()
        u, b, c = sequence.float(), b.float(), c.float()
        use_kernel = self.backend == "mamba" or (self.backend == "auto" and u.is_cuda)
        if use_kernel:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            result = selective_scan_fn(
                u.transpose(1, 2).contiguous(),
                dt.transpose(1, 2).contiguous(),
                a,
                b.transpose(1, 2).contiguous(),
                c.transpose(1, 2).contiguous(),
                self.skip[direction].float(),
                delta_softplus=False,
            ).transpose(1, 2)
        else:
            state = u.new_zeros(u.shape[0], u.shape[2], self.state_dim)
            values = []
            for t in range(u.shape[1]):
                step = dt[:, t, :, None]
                state = (step * a).exp() * state + step * b[:, t, None, :] * u[
                    :, t, :, None
                ]
                values.append(
                    (state * c[:, t, None, :]).sum(-1) + self.skip[direction] * u[:, t]
                )
            result = torch.stack(values, dim=1)
        return result.to(sequence.dtype)

    def forward(self, x):
        batch, height, width, channels = x.shape
        row = x.reshape(batch, height * width, channels)
        col = x.transpose(1, 2).reshape(batch, height * width, channels)
        sequences = (row, col, row.flip(1), col.flip(1))
        outputs = [self.scan(seq, i) for i, seq in enumerate(sequences)]
        row_result = outputs[0] + outputs[2].flip(1)
        col_result = outputs[1] + outputs[3].flip(1)
        col_result = col_result.reshape(batch, width, height, channels).transpose(1, 2)
        return self.norm(row_result.reshape_as(x) + col_result)


class SCIM(nn.Module):
    def __init__(self, channels, hidden_dim, state_dim=16, backend="auto"):
        super().__init__()
        self.norm = nn.LayerNorm(2 * channels)
        self.gate = nn.Linear(2 * channels, hidden_dim)
        self.state = nn.Linear(2 * channels, hidden_dim)
        self.depthwise = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim
        )
        self.scan = SelectiveScan2D(hidden_dim, state_dim, backend)
        self.projection = nn.Linear(hidden_dim, 2 * channels)

    def forward(self, rgb, xyz):
        original = torch.cat((rgb, xyz), dim=1).permute(0, 2, 3, 1)
        x = self.norm(original)
        state = self.depthwise(self.state(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        update = self.projection(self.scan(state) * F.silu(self.gate(x)))
        return (original + update).permute(0, 3, 1, 2)


class SSRL(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        hidden_dim=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        mc_samples=4,
        mc_dropout=0.1,
        state_dim=16,
        scan_backend="auto",
        clip_model="ViT-B/32",
        text_embeddings=None,
        feature_jitter=None,
    ):
        super().__init__()
        if len(inplanes) != 1 or len(instrides) != 1:
            raise ValueError("SSRL expects one aligned feature scale per modality")
        if mc_samples < 2 or not 0 < mc_dropout < 1:
            raise ValueError(
                "uncertainty estimation requires mc_samples >= 2 and 0 < mc_dropout < 1"
            )
        self.channels = inplanes[0]
        self.stride = instrides[0]
        self.feature_size = tuple(feature_size)
        self.mc_samples, self.mc_dropout = mc_samples, mc_dropout
        self.feature_jitter = feature_jitter
        count = math.prod(feature_size)
        self.anchors = SemanticAnchors(hidden_dim, nhead, clip_model, text_embeddings)
        self.rgb_proj = nn.Linear(self.channels, hidden_dim)
        self.xyz_proj = nn.Linear(self.channels, hidden_dim)
        self.position = nn.Parameter(torch.randn(count, 1, hidden_dim) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, 1, hidden_dim) * 0.02)
        self.rgb_queries = nn.Parameter(torch.randn(count, 1, hidden_dim) * 0.02)
        self.xyz_queries = nn.Parameter(torch.randn(count, 1, hidden_dim) * 0.02)
        self.decoder_positions = nn.Parameter(torch.randn(2, count, 1, hidden_dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout),
            num_encoder_layers,
        )
        self.decoder_rgb = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout),
            num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.decoder_xyz = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout),
            num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.rgb_output = nn.Linear(hidden_dim, self.channels)
        self.xyz_output = nn.Linear(hidden_dim, self.channels)
        self.scim = SCIM(self.channels, hidden_dim, state_dim, scan_backend)
        self.semantic_projection = nn.Linear(hidden_dim, self.channels)
        self.reliability = nn.Conv2d(2, 1, 1)

    def get_outplanes(self):
        return [2 * self.channels]

    def get_outstrides(self):
        return [self.stride]

    def forward(self, inputs):
        rgb, xyz = inputs["feature_align"], inputs["feature_align_xyz"]
        if rgb.shape != xyz.shape or tuple(rgb.shape[-2:]) != self.feature_size:
            raise ValueError(
                "RGB and XYZ aligned features must match configured spatial dimensions"
            )
        batch, _, height, width = rgb.shape
        z = self.anchors(batch)
        rgb_tokens = self.rgb_proj(rgb.flatten(2).permute(2, 0, 1))
        xyz_tokens = self.xyz_proj(xyz.flatten(2).permute(2, 0, 1))
        tokens = torch.cat(
            (
                rgb_tokens + self.position + self.modality[0],
                xyz_tokens + self.position + self.modality[1],
                z,
            ),
            dim=0,
        )
        memory = self.encoder(tokens)
        h_rgb = self.decoder_rgb(
            (self.rgb_queries + self.decoder_positions[0]).expand(-1, batch, -1),
            memory,
        )
        h_xyz = self.decoder_xyz(
            (self.xyz_queries + self.decoder_positions[1]).expand(-1, batch, -1),
            memory,
        )

        def spatial(x):
            return x.permute(1, 2, 0).reshape(batch, -1, height, width)

        rec_rgb, rec_xyz = spatial(self.rgb_output(h_rgb)), spatial(
            self.xyz_output(h_xyz)
        )        if self.training:
            samples = torch.stack(
                [
                    spatial(
                        self.xyz_output(
                            F.dropout(h_xyz, p=self.mc_dropout, training=True)
                        )
                    )
                    for _ in range(self.mc_samples)
                ]
            )
            uncertainty = samples.var(dim=0, unbiased=False).mean(1, keepdim=True)
        else:            variance = F.linear(
                h_xyz.square() * self.mc_dropout / (1 - self.mc_dropout),
                self.xyz_output.weight.square(),
            )
            uncertainty = spatial(variance).mean(1, keepdim=True)
        semantic = self.semantic_projection(z.mean(0))[:, :, None, None]
        consistency = F.cosine_similarity(rec_xyz, semantic, dim=1).unsqueeze(1)
        rho = torch.sigmoid(
            self.reliability(torch.cat((uncertainty, consistency), dim=1))
        )
        fusion = self.scim(rec_rgb, rec_xyz)
        final = fusion + torch.cat((torch.zeros_like(rec_xyz), rho * rec_xyz), dim=1)
        target = torch.cat((rgb, xyz), dim=1)
        pred = torch.linalg.vector_norm(final - target, dim=1, keepdim=True)
        size = (
            inputs["image"].shape[-2:]
            if "image" in inputs
            else (height * self.stride, width * self.stride)
        )
        return {
            "feature_rec": rec_rgb,
            "feature_rec_xyz": rec_xyz,
            "feature_align": rgb,
            "feature_align_xyz": xyz,
            "feature_rec_final": final,
            "feature_target": target,
            "topology_rgb": spatial(h_rgb),
            "topology_xyz": spatial(h_xyz),
            "representation": final,
            "uncertainty": uncertainty,
            "reliability": rho,
            "pred": F.interpolate(
                pred, size=size, mode="bilinear", align_corners=False
            ),
        }
