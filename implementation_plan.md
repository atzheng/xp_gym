# Goals

We are trying to reproduce qualitatively the ridesharing experiments in the paper: Markovian interference in experiments https://openreview.net/forum?id=AOSIbSmQJr

The paper introduces an estimator for average treatment effects under temporal interference.

The UberPool setting used in the paper introduces a particular type of temporal interference. The experiment uses A/B testing at the request level to compare two policies: one which is more likely to assign a request to a vehicle that already has a rider in it (Pool), and one which is more likely to assign a request to an empty vehicle. Allocating riders to a pool vehicle will generally be cheaper in the short run than allocating them an empty vehicle, since as long as there's some overlap between the trips, it'll be cheaper to serve them both using the same vehicle. But, since it uses up capacity in a matched vehicle, that increases costs downstream. Likewise, allocating a request to an empty vehicle, creates a Pool vehicle which can matching to new requests downstream, thereby lowering the cost of serving that future request. The interference comes from the fact that a naive AB test will only be able to measure the difference in cost at time of matching between two policies, and will ignore the downstream effects.


# Commands

The basic command to compute the ate for a given configuration:

`python scripts/compute-ate.py env=rideshare_pool env_params=rideshare_pool`

And to run the estimators for a given configuration:

`python scripts/compute-ate.py env=rideshare_pool env_params=rideshare_pool`

# Tasks

## Task 1: demonstrate a problem setting under which the truncated DQ estimator outperforms Naive on this problem.

The main lever that we have currently over the two policies being tested, is the *savings threshold*. For each request, we compute the cost of the best pool dispatch and the cost of the best empty car dispatch, and we only dispatch the pool car if the cost savings relative to the empty car are above the *savings threshold*. Raising the threshold results in fewer pool dispatches, thereby increasing short-term costs relative to long-term costs.


## Task 2: Develop a version of the LSTD DQ estimator that achieves even better performance. Some routes to pursue:

The DQ estimator that we are dealing with here is using average reward LSTD to estimate Q values. In contrast, the truncated DQ estimator uses truncated Monte Carlo estimate of the Q values. We know that this version works. The question is, how can we get the version with function approximation to work as well?

1. As a means of reducing variance: for DQ, only use timesteps where policy A would have differed from policy B.
2. Find some way to interpolate between the truncated Monte-Carlo DQ estimator and the LSTD version.
3. Try different 


