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
def single_exponential_kernel():
    support_range = torch.arange(0, 100, 1).unsqueeze(1)
    tau = torch.tensor([10.]).unsqueeze(0)

    basis = torch.exp(-support_range / tau)

    ker = Kernel(basis=basis)

    return ker


@pytest.fixture
def glm(exponential_basis_kernel):
    ker = exponential_basis_kernel
    glm = GLM(bias=-3.5, stim_kernel=ker.clone(), hist_kernel=ker.clone())
    return glm


@pytest.fixture
def simple_glm(single_exponential_kernel):
    stim_ker = single_exponential_kernel.clone()
    stim_ker.weight = torch.nn.Parameter(torch.tensor([0.3]))
    hist_ker = single_exponential_kernel.clone()
    hist_ker.weight = torch.nn.Parameter(torch.tensor([-3.]))
    glm = GLM(bias=-3.3, stim_kernel=stim_ker.clone(), hist_kernel=hist_ker.clone())
    return glm


def test_conditional_intensity(glm):
    dt = 1
    t = torch.arange(0, 500, dt)  # Time points
    n = 3  # Number of samples
    stim = torch.randn(len(t), 2)

    log_lam, mask_spikes = glm.sample(t, stim=stim, shape=(n,))
    log_lam_cond = glm.log_conditional_intensity(t, mask_spikes, stim=stim)

    assert torch.all(torch.isclose(log_lam, log_lam_cond, atol=1e-5))


def test_fitting_recovers_true_glm(simple_glm):
    glm_true = simple_glm.clone()
    glm = GLM(bias=-5.,
              stim_kernel=Kernel(basis=glm_true.stim_kernel.basis),
              hist_kernel=Kernel(basis=glm_true.hist_kernel.basis))
    dt = 1
    t = torch.arange(0, 10000, dt)  # Time points
    n = 3  # Number of samples
    stim = torch.randn(len(t), 2)

    _, mask_spikes = glm_true.sample(t, stim=stim, shape=(n,))

    optim = torch.optim.Adam(glm.parameters(), lr=5e-3, betas=(0, 0.9))

    glm.fit(t, mask_spikes, stim=stim, num_epochs=1500, optim=optim, verbose=False)

    assert torch.all(torch.isclose(glm.bias, glm_true.bias, rtol=1e-1))
    assert torch.all(torch.isclose(glm.stim_kernel.weight, glm_true.stim_kernel.weight, rtol=1e-1))
    assert torch.all(torch.isclose(glm.hist_kernel.weight, glm_true.hist_kernel.weight, rtol=1e-1))


if __name__ == '__main__':
    pytest.main()
