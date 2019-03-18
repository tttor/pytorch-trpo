# Fact
* use filter state
```
running_state = ZFilter((num_inputs,), clip=5)
```
* use filter reward with demean=False
  * but this `running_reward` is NEVER been used
```
running_reward = ZFilter((1,), demean=False, clip=10)
```

* NOT reset env after an episode ends (done is TRUE)

* KLdiv: sum over action dimensions, then average over states in a given batch
```
return kl.sum(dim=1, keepdim=True) # sum over action dimensions
```

* train in one epoch, and fullbatch

* damping in Fvp()

* use scipy.optimize.fmin_l_bfgs_b()

* line search
  * which is different from the paper, so how come?
  * but same with /home/tor/ws-tmp/modular_rl/modular_rl/trpo.py
```
    actual_improve = fval - newfval
    expected_improve = expected_improve_rate * stepfrac
    ratio = actual_improve / expected_improve
    if ratio.item() > accept_ratio and actual_improve.item() > 0:
        return True, xnew
return False, x
```
