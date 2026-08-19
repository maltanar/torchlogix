import torch


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """

    def __init__(self, k: int, tau: float = 1.0, beta=0.0, device="cpu", export_mode=False):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta
        self.device = device
        self.export_mode = export_mode

    def forward(self, x):
        assert x.shape[-1] % self.k == 0, "The number of input features must be divisible by k."

        if torch.onnx.is_in_onnx_export() and x.dtype in (torch.bool, torch.uint8):
            # LookupTable emits uint8; widen to avoid overflow in the reduction
            x = x.to(torch.int32)

        result = x.reshape(x.shape[:-1] + (self.k, x.shape[-1] // self.k)).sum(-1)
        if self.beta != 0.0:
            result = result + self.beta
        if self.tau != 1.0:
            result = result / self.tau
        return result

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)

    def set_export_mode(self, export_mode: bool):
        self.eval()
        self.export_mode = export_mode
