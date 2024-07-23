// Simple Q learning model
// Based on the bandit2arm_delta.stan model from hBayesDM.
// Changes:
// 1. Data is used in long format, rather than wide format utilized by hBayesDM.
// 2. Initial Q values given as data

data {
  int<lower=1> N; // Total number of trials
  int<lower=1> N_bl; // Number of blocks
  array[N] int<lower=1, upper=N_bl> bl; // Block number
  array[N] int<lower=0, upper=1> choice; // Binomial choice, coded as 1 for optimal stimulus, 0 for other stimulus 
  vector[N] outcome;  // Outcome observed, unbounded
  vector[2] initV;  // initial values for EV
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Population-parameters
  real a; // Learning rate on unconstrained scale

  real rho; // Reward sensitivity
}

transformed parameters {
  vector[2] initVr = initV .* rho; // Initial values scaled by rho
}

model {
  // Transform learning rate
  real alpha = Phi_approx(a);

  // Priors
  a  ~ normal(0, 1);
  rho  ~ normal(0, 1);

  // Loop over trials, updating EVs and incrementing log-density
  {    
    real PE;      // prediction error
    vector[2] ev = initVr; // expected value
 
    for (i in 1:N) {

      // Set ev to initial values if participant/block different from previous trial
      if ((i > 1) && (bl[i] != bl[i-1]))
        ev = initVr;

      // Increment log densitiy
      choice[i] ~ bernoulli_logit(ev[2] - ev[1]);

      // prediction error
      PE = outcome[i] * rho - ev[choice[i] + 1];

      // value updating (learning)
      ev[choice[i] + 1] += alpha * PE;
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=1> alpha = Phi_approx(a);
}
