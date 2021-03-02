import torch


class PositionalEncoding:
    @staticmethod
    def get_matrix(input_tensor_size) -> torch.Tensor:
        """
        :param input_tensor_size: size of feature map to be applied with PositionalEncoding
        :return:
            PositionalEncoding Matrix
        """

        h, w, c = input_tensor_size

        if c % 4 != 0:
            raise ValueError("incorrect channel dimension for PE matrix")
        PE = torch.ones(h, w, c)

        h_depended, w_depended = PositionalEncoding.__get_mesh_grid(h, w)

        for pe_channel in range(c // 4):

            channel_norm = 10000 ** (2 * pe_channel // c)

            PE[:, :, pe_channel] = torch.sin((h_depended * 256) / (h * channel_norm))
            PE[:, :, pe_channel + 1] = torch.cos(
                (h_depended * 256) / (h * channel_norm)
            )
            PE[:, :, pe_channel + 2] = torch.sin(
                (w_depended * 256) / (w * channel_norm)
            )
            PE[:, :, pe_channel + 3] = torch.cos(
                (w_depended * 256) / (w * channel_norm)
            )

        return PE

    @staticmethod
    def __get_mesh_grid(h, w):
        h_arranged = torch.arange(h, dtype=torch.float32)
        w_arranged = torch.arange(w, dtype=torch.float32)

        h_depended, w_depended = torch.meshgrid(h_arranged, w_arranged)

        return h_depended, w_depended
