from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, mask):
        # x shape: (B, H*W, v_in_channels) vis
        # l input shape: (B, l_in_channels, N_l) text
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        mask = mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(l)  # (B, self.key_channels, N_l)
        key = self.f_key(x)
        key = key.permute(0, 2, 1)  # (B, HW, key_channels)
        value = self.f_value(x)
        value = value.permute(0, 2, 1)  # (B, HW, key_channels)
        query = query * mask  # (B, key_channels, N_l)
        n_l = query.size(-1)

        query = query.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l).permute(0, 1, 3, 2)
        # (b, num_heads, n_l, self.key_channels//self.num_heads)
        key = key.reshape(B, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 3, 1)
        # (b, num_heads, self.key_channels//self.num_heads, H*W)
        value = value.reshape(B, HW, self.num_heads, self.value_channels // self.num_heads).permute(0, 2, 3, 1)
        # (b, num_heads, self.key_channels//self.num_heads, H*W)

        mask = mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key).permute(0, 1, 3, 2)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4 * mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1).permute(0, 1, 3, 2)  # (B, num_heads, N_l, h*w)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, N_l, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, n_l, self.value_channels)  # (B, N_l, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, NL)
        out = self.W(out)  # (B, value_channels, NL)
        out = out.permute(0, 2, 1)  # (B, NL, value_channels)

        return out


class TextResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, vis_dim: int = 768, n_head: int = 1, attn_mask: torch.Tensor = None):
        super(TextResidualAttentionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # input x shape: (B, NL, dim)
        dropout = 0.5
        self.vis_project = nn.Sequential(nn.Conv1d(vis_dim, vis_dim, 1, 1),
                                         # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                         )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels=d_model,  # v_in
                                                            l_in_channels=vis_dim,  # l_in
                                                            key_channels=d_model,  # key
                                                            value_channels=d_model,  # value
                                                            out_channels=d_model,  # out
                                                            num_heads=n_head
                                                            )

        self.project_mm = nn.Sequential(nn.Conv1d(d_model, d_model, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, text, image, text_mask):
        # print(text.size())  # 77, 1, 512  NL,B,dim
        # print(image.size())  # 1, 50, 768

        # input x shape: (B, H*W, dim)
        input_text = text.permute(1, 2, 0)  # B, dim, NL
        vis = self.vis_project(image.permute(0, 2, 1)).permute(0, 2, 1)  # (B, dim, H*W)
        lang = self.image_lang_att(vis, input_text, text_mask)  # (B, NL, dim)  # expect text=1, 512, 77
        lang = lang.permute(0, 2, 1)  # (B, dim, NL)
        mm = torch.mul(input_text, lang)
        mm = self.project_mm(mm)  # (B, dim, NL)
        mm = mm.permute(2, 0, 1)  # (B, NL, dim)
        mm = self.ln_1(mm)

        text = text + self.attention(self.ln_1(text))
        text = text + self.mlp(self.ln_2(text))

        # result_text = mm + text

        # expect text = 77, 1, 512
        return mm, image


class TextTransformer(nn.Module):
    def __init__(self, width: int, vis_dim: int, layer_num: int, heads: int, attn_mask: torch.Tensor = None):
        super(TextTransformer, self).__init__()
        self.width = width
        self.layer_num = layer_num
        # self.resblocks = nn.Sequential(
        #     *[TextResidualAttentionBlock(width, heads, attn_mask) for _ in range(self.layer_num)])

        self.resblocks = nn.ModuleList([TextResidualAttentionBlock(width, vis_dim=vis_dim, n_head=heads, attn_mask=attn_mask) for _ in range(self.layer_num)])
        # for i_layer in range(self.layer_num):
        #     layer = TextResidualAttentionBlock(width, vis_dim=vis_dim, n_head=heads, attn_mask=attn_mask)
        #     self.resblocks.append(layer)

    def forward(self, x: torch.Tensor, image: torch.Tensor = None, text_mask: torch.Tensor = None):
        text = x
        for index, layer in enumerate(self.resblocks):
            text, image = layer(text, image, text_mask)

        return text
