import pytest
import torch

from kernel.base import Kernel
from mmdglm.glm.base import GLM


@pytest.fixture
def exponential_basis_kernel():
    support_range = torch.arange(0, 500, 1).unsqueeze(1)
    tau = torch.tensor([10., 50]).unsqueeze(0)

    basis = torch.exp(-support_range / tau)
    weight = torch.tensor([-5., 1])

    ker = Kernel(basis=basis,
                 weight=weight)

    return ker


@pytest.fixture
def glm(exponential_basis_kernel):
    ker = exponential_basis_kernel
    glm = GLM(bias=-3.5, stim_kernel=ker.clone(), hist_kernel=ker.clone())
    return glm


def test_conditional_intensity(glm):
    dt = 1
    t = torch.arange(0, 500, dt)  # Time points
    n = 3  # Number of samples
    stim = torch.randn(len(t), 2)

    log_lam, mask_spikes = glm.sample(t, stim=stim, shape=(n,))
    log_lam_cond = glm.log_conditional_intensity(t, mask_spikes, stim=stim)

    assert torch.all(torch.isclose(log_lam, log_lam_cond, atol=1e-5))


if __name__ == '__main__':
    pytest.main()
