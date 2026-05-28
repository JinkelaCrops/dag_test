import torch
import numpy as np
import scipy.linalg as slin
import notears.cupy_expm.cupy_extension as cpe
import cupy


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        t = torch.as_tensor(f, dtype=input.dtype)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


class TraceExpmGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        c = cupy.asarray(input.detach())
        E = cpe.expm(c)
        f = cupy.trace(E)
        E = torch.from_dlpack(E)
        ctx.save_for_backward(E)
        t = torch.from_dlpack(f)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply
trace_expm_gpu = TraceExpmGPU.apply


def main():
    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(trace_expm, input)

    input = torch.tensor([[1, 2], [3, 4.]], requires_grad=True)
    tre = trace_expm(input)
    f = 0.5 * tre * tre
    print('f\n', f.item())
    f.backward()
    print('grad\n', input.grad)

    input2 = torch.tensor([[1, 2], [3, 4.]], requires_grad=True, device='cuda:0')
    tre = trace_expm_gpu(input2)
    f = 0.5 * tre * tre
    print('f\n', f.item())
    f.backward()
    print('grad\n', input2.grad)


if __name__ == '__main__':
    # import numpy as np
    # import notears.cupy_expm.cupy_extension as cpe
    # import torch as t
    # import cupy
    #
    # a = np.random.rand(2, 2)
    # a1 = t.from_numpy(a).to('cuda:0')
    # a2 = cupy.asarray(a1)
    # a3 = cpe.expm(a2)
    # a4 = t.as_tensor(a3)

    main()
