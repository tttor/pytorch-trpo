import numpy as np
import torch
from torch.autograd import Variable
from utils import *

def trpo_step(model, get_loss, get_kl, max_kl, damping):
    def Fvp(v):
        kl = get_kl().mean()
        grad_kl = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_kl_v = (grad_kl * Variable(v)).sum()
        grad_grad_kl_v = torch.autograd.grad(grad_kl_v, model.parameters())
        grad_grad_kl_v = torch.cat([grad.contiguous().view(-1) for grad in grad_grad_kl_v]).data
        return grad_grad_kl_v + (v*damping)

    # Compute loss and grad
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    # Step dir
    # Using linear approximation to the objective fn, but how does it look like exactly?
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    # Step len
    sFs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(sFs / max_kl) # lm for "lagrange multiplier"
    fullstep = stepdir / lm[0]
    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])

    # Update
    set_flat_params_to(model, new_params)
    return loss

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    fval = f(True).data
    # print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("fval after", newfval.item())
            return True, xnew
    return False, x
