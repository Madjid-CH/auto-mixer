
import torch
import torch.nn.functional as F


class ConcatFusion:
    def __init__(self, dim=1, **_kwargs):
        self.dim = dim

    def __call__(self, *args):
        args = pad_tensors(args)
        return torch.cat(args, dim=self.dim)

    def get_output_shape(self, *args, dim=None):
        """
        Returns the output shape of the layer given the input shape.
        Parameters
        ----------
        *args : tuple, list, torch.Size, int
            The input shape of the layer. If a tuple, list, or torch.Size, then the full shape is expected. If an int,
            then the dimension parameter is also expected, and the result will be the output shape of that dimension.
        dim : int, optional
            The dimension of the input shape. Only used if the first argument is an int. Defaults to None. If not None,
            then the args argument is expected to be an int and match the input shape of at the given dimension.

        Returns
        -------
        tuple, int

        """
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
            if dim == self.dim:
                return sum(args)
            else:
                return args[0]
        shape = list(args[0])
        for arg in args[1:]:
            shape[self.dim] += arg[self.dim]
        return tuple(shape)


def pad_tensors(args):
    max_size = max(arg.size(2) for arg in args)
    padded_tensors = []
    for tensor in args:
        padding = (0, max_size - tensor.size(2))
        padded_tensors.append(F.pad(tensor, padding))
    return padded_tensors


class MaxFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.maximum(*args)

    @staticmethod
    def get_output_shape(*args, dim=None):
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
        if args[0] != args[1]:
            raise ValueError("Input shapes must be equal")
        return args[0]


class SumFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.add(*args)

    @staticmethod
    def get_output_shape(*args, dim=None, **kwargs):
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
        if args[0] != args[1]:
            raise ValueError("Input shapes must be equal")
        return args[0]


class MeanFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.mean(torch.stack(args), 0)

    @staticmethod
    def get_output_shape(*args, dim=None, **_kwargs):
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
        if args[0] != args[1]:
            raise ValueError("Input shapes must be equal")
        return args[0]
