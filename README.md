# Fact
* use filter state and reward
```
running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
```
* not reset env after an episode ends (done is TRUE)
* KLdiv: sum over action dimensions, then average over states in a given batch
```
return kl.sum(dim=1, keepdim=True) # sum over action dimensions
```
* use scipy.optimize.fmin_l_bfgs_b()


