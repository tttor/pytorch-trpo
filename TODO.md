# TODO.md

# Question
* damping in Fvp() in trpo_step()
* shs stands for?
  * ANS: it is for `s^T H s`,
    where s is the step dir and H is some notion of Hessian, eg the Fisher info matrix;
    see Appendix C

```
# Step len
shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
```


# /home/tor/ws-tmp/stable-baselines/stable_baselines/trpo_mpi/trpo_mpi.py
* aka: /home/tor/ws-tmp/baselines/baselines/trpo_mpi/trpo_mpi.py

* step size is via
```
stepsize = 1.0
thbefore = self.get_flat()
thnew = None
for _ in range(10):
    thnew = thbefore + fullstep * stepsize
    self.set_from_flat(thnew)
    mean_losses = surr, kl_loss, *_ = self.allmean(
        np.array(self.compute_losses(*args, sess=self.sess)))
    improve = surr - surrbefore
    logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
    if not np.isfinite(mean_losses).all():
        logger.log("Got non-finite value of losses -- bad!")
    elif kl_loss > self.max_kl * 1.5:
        logger.log("violated KL constraint. shrinking step.")
    elif improve < 0:
        logger.log("surrogate didn't improve. shrinking step.")
    else:
        logger.log("Stepsize OK!")
        break
    stepsize *= .5```

* how does lagrange multplier work here?
```
shs = .5*stepdir.dot(fisher_vector_product(stepdir))
lm = np.sqrt(shs / max_kl)
# logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
fullstep = stepdir / lm
```
