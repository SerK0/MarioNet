import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        stride = 1 if not downsample else 2

        if self.downsample:
            self.identity_sparse = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.identity_sparse = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        intermediate_res = self.act1(self.bn1(self.conv1(x)))
        final_res = self.bn2(self.conv2(intermediate_res))

        return self.act(final_res + self.identity_sparse(x))


class DriverEncoder(nn.Module):
    def __init__(self, config_driver_encoder, feature_dim):
        super(DriverEncoder, self).__init__()

        self.config = config_driver_encoder

        input_feature_dim = config_driver_encoder['input_feature_dim']
        hidden_features_dim = config_driver_encoder['hidden_features_dim']

        assert config_driver_encoder['depth'] == len(config_driver_encoder['hidden_features_dim']), \
                "inconsistent depth of driver encoder and len of hidden dims"

        self.block1 = BasicBlock(input_feature_dim, hidden_features_dim[0], downsample=True)

        self.blocks = nn.ModuleList([
            BasicBlock(hidden_features_dim[idx], hidden_features_dim[idx + 1], downsample=True) \
                                for idx, hidden_dim in enumerate(hidden_features_dim[:-1])
        ])

        self.block1 = BasicBlock(feature_dim, feature_dim * 2, downsample=True)
        self.block2 = BasicBlock(feature_dim * 2, feature_dim * 4, downsample=True)
        self.block3 = BasicBlock(feature_dim * 4, feature_dim * 8, downsample=True)
        self.block4 = BasicBlock(feature_dim * 8, feature_dim * 16, downsample=True)

    def forward(self, rx):
        x = self.block1(rx)
        x = self.block2(x)
        x = self.block3(x)
        zx = self.block4(x)

        return zx


class PositionalEncoding:
    @staticmethod
    def get_matrix(input_tensor_size) -> torch.Tensor:
        '''
        :param input_tensor_size: size of feature map to be applied with PositionalEncoding
        :return:
            PositionalEncoding Matrix
        '''

        h, w, c = input_tensor_size

        if c % 4 != 0:
            raise ValueError("incorrect channel dimension for PE matrix")
        PE = torch.ones(h, w, c)

        h_depended, w_depended = PositionalEncoding.__get_mesh_grid(h, w)

        for pe_channel in range(c // 4):

            channel_norm = 10000 ** (2 * pe_channel // c)

            PE[:, :, pe_channel] = torch.sin((h_depended * 256) / (h * channel_norm))
            PE[:, :, pe_channel + 1] = torch.cos((h_depended * 256) / (h * channel_norm))
            PE[:, :, pe_channel + 2] = torch.sin((w_depended * 256) / (w * channel_norm))
            PE[:, :, pe_channel + 3] = torch.cos((w_depended * 256) / (w * channel_norm))

        return PE

    @staticmethod
    def __get_mesh_grid(h, w):
        h_arranged = torch.arange(h, dtype=torch.float32)
        w_arranged = torch.arange(w, dtype=torch.float32)

        h_depended, w_depended = torch.meshgrid(h_arranged, w_arranged)

        return h_depended, w_depended


class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 driver_feature_dim,
                 target_feature_dim,
                 attention_feature_dim
                ):
        super(SelfAttentionBlock, self).__init__()

        self.q_proj = nn.Linear(driver_feature_dim, attention_feature_dim)
        self.px_proj = nn.Linear(driver_feature_dim, attention_feature_dim)

        self.k_proj = nn.Linear(target_feature_dim, attention_feature_dim)
        self.py_proj = nn.Linear(target_feature_dim, attention_feature_dim)

        self.v_proj = nn.Linear(target_feature_dim, attention_feature_dim)

        self.attention_feature_size = attention_feature_dim

    def forward(self, zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor[torch.float32]:
        '''
        :param zx: driver feature map tensor --- size: [B x cx x H x W]
        :param zy: target feature map tensor --- size: [B x K x cy x H x W]
        :return:
            self 'attentioned' feature map
        '''
        batch_size, cx, hx, wx = zx.size()
        Px = torch.cat([PositionalEncoding.get_matrix((hx, wx, cx)).unsqueeze(0) for _ in range(batch_size)], dim=0)
        q = self.q_proj(zx.permute(0, 2, 3, 1)) + self.px_proj(Px)

        batch_size, K, cy, h, w = zy.size()

        Py = torch.cat([PositionalEncoding.get_matrix((h, w, cy)).unsqueeze(0) for _ in range(batch_size * K)], dim=0)
        Py = Py.view(batch_size, K, h, w, cy)

        k = self.k_proj(zy.permute(0, 1, 3, 4, 2)) + self.py_proj(Py)

        v = self.v_proj(zy.permute(0, 1, 3, 4, 2))

        q_flatten = q.view(batch_size, -1, self.attention_feature_size)
        k_flatten = k.view(batch_size, -1, self.attention_feature_size)

        attn_value = torch.bmm(q_flatten, k_flatten.T) / torch.sqrt(self.attention_feature_size)
        orig_shape = attn_value.size()

        softmax_attentioned = F.softmax(attn_value.view(batch_size, -1)).view(*orig_shape)
        output_t = torch.bmm(softmax_attentioned, v.view(batch_size, -1, self.attention_feature_size))

        return output_t.view(batch_size, cx, hx, wx)


class Blender(nn.Module):
    def __init__(self, config_blender):
        super(Blender, self).__init__()

        self.self_attnblock = SelfAttentionBlock(
            config_blender['driver_feature_dim'],
            config_blender['target_feature_dim'],
            config_blender['attention_feature_dim']
        )

        self.inst_norm1 = nn.InstanceNorm2d(config_blender['driver_feature_dim'])

        self.conv = nn.Conv2d(config_blender['driver_feature_dim'],
                              config_blender['driver_feature_dim'],
                              kernel_size=3,
                              padding=1)

        self.inst_norm2 = nn.InstanceNorm2d(config_blender['driver_feature_dim'])

    def forward(self, zx, Zy):
        mixed_feature = self.self_attnblock(zx, Zy)
        normed = self.inst_norm1(mixed_feature)
        return self.inst_norm2(normed + self.conv(normed))
