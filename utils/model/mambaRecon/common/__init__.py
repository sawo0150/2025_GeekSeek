# mambaRecon/common/__init__.py
import torch, torch.nn as nn

# def repeat(x, pattern, **sizes):
#     B, G, N, L = x.shape
#     H = sizes["H"]
#     return x.reshape(B, G, 1, N, L).expand(B, G, H, N, L).reshape(B, G*H, N, L)

def repeat(x, pattern, **sizes):
    """
    Simplified repeat utility supporting:
      - 1D tensor repetition (e.g. 'n -> d n' or 'n1 -> r n1').
      - 4D tensor repetition for patchify: (B, G, N, L) -> (B, G*H, N, L).
    """
    # 1D repeat: A_log_init, D_init 용
    if x.ndim == 1 and "->" in pattern:
        # sizes 에 반복 횟수 하나만 들어있다고 가정 (예: d 또는 r)
        reps = next(iter(sizes.values()))
        # (reps, x_length) 로 확장
        return x.unsqueeze(0).expand(reps, x.size(0)).contiguous()

    # 4D repeat: 기존 패치용
    if x.ndim == 4:
        B, G, N, L = x.shape
        H = sizes.get("H")
        if H is None:
            raise ValueError(f"repeat: missing 'H' for 4D input, got sizes={sizes}")
        return x.reshape(B, G, 1, N, L) \
                .expand(B, G, H, N, L) \
                .reshape(B, G * H, N, L)

    raise ValueError(f"repeat: unsupported input shape {tuple(x.shape)} for pattern '{pattern}'")

def rearrange(x, pattern, **sizes):
    if pattern.startswith('b h w (p1 p2 c)'):
        p1, p2, c = sizes["p1"], sizes["p2"], sizes["c"]
        B, H, W, C = x.shape
        x = x.view(B, H, W, p1, p2, c).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, H*p1, W*p2, c)
    raise NotImplementedError(pattern)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: 
            return x
        keep = 1. - self.drop_prob
        mask = torch.rand(x.size(0), *[1]*(x.dim()-1), device=x.device) < keep
        return x.div(keep) * mask
