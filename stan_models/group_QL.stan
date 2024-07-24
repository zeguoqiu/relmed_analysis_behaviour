// Simple Q learning model
// Based on the bandit2arm_delta.stan model from hBayesDM.
// Changes:
// 1. Data is used in long format, rather than wide format utilized by hBayesDM.
// 2. Initial Q values given as data

data {
  int<lower=1> N; // Total number of trials
  int<lower=1> N_p; // Number of participants
  int<lower=1> N_bl; // Number of blocks
  array[N] int<lower=1, upper=N_p> pp; // Participant number for each trial
  array[N] int<lower=1, upper=N_bl> bl; // Block number
  array[N] int<lower=0, upper=1> choice; // Binomial choice, coded as 1 for optimal stimulus, 0 for other stimulus 
  vector[N] outcome;  // Outcome observed, unbounded
  vector[2] initV;  // initial values for EV
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Population-parameters
  real mu_a; // Average learning rate on unconstrained scale
  real<lower=0> sigma_a; // Standard deviation of learning rate on unconstrained scale

  real mu_rho; // Average reward sensitivy
  real<lower=0> sigma_rho; // Standard deviation of reward sensitivy

  // Subject-level deviations
  vector[N_p] a;    // standardized learning rate deviations
  vector[N_p] r;  // standardized reward sensitivity deviations
}

model {
  // Initiate subject-level parameters
  vector[N_p] alpha;
  vector[N_p] rho;

  // Compute subject-level parameters
  alpha = Phi_approx(mu_a  + sigma_a  * a);
  rho = mu_rho + sigma_rho * r;

  // Population level parameters
  mu_a  ~ normal(0, 1);
  mu_rho  ~ normal(0, 1);
  sigma_a ~ normal(0, 0.2);
  sigma_rho ~ normal(0, 0.2);

  // individual parameters
  a ~ normal(0, 1);
  r ~ normal(0, 1);

  // Loop over trials, updating EVs and incrementing log-density
  {    
    real PE;      // prediction error
    vector[2] ev; // expected value
 
    for (i in 1:N) {

      // Set ev to initial values if participant/block different from previous trial
      if ((i == 1) || ((pp[i] != pp[i-1]) || (bl[i] != bl[i-1])))
        ev = initV .* rho[pp[i]];

      // Increment log densitiy
      choice[i] ~ bernoulli_logit(ev[2] - ev[1]);

      // prediction error
      PE = outcome[i] * rho[pp[i]] - ev[choice[i] + 1];

      // value updating (learning)
      ev[choice[i] + 1] += alpha[pp[i]] * PE;
    }
  }
}
