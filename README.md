# Fact
* use filter state and reward
```
running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
```
* not reset env after an episode ends (done is TRUE)
* use scipy.optimize.fmin_l_bfgs_b()


