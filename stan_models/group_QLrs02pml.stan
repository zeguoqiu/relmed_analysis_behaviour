// Simple Q learning model
// Based on the bandit2arm_delta.stan model from hBayesDM.
// Changes:
// 1. Data is used in long format, rather than wide format utilized by hBayesDM.
// 2. Initial Q values given as data
// 3. Using reduce_sum to parallelize inference

functions{
  real partial_sum(
    array [] vector z, // By participant parameters: rho, alpha
    int start,
    int end,
    array[] int pp,
    array[] int p1t,
    array[] int bl,
    array[] int valence,
    array[] int choice,
    vector outcome,
    real aao
  )
  {
    // Compute trial-wise indices for this subset of participants
    int dstart = p1t[start]; // First trial in this subset
    int dend = p1t[end+1] - 1; // Last trial in this subset
    int N = dend - dstart + 1; // n trials in this subset

    // Subset trial-wise inputs (t is for this)
    array[N] int tpp = pp[dstart:dend]; // Uncorrected participant indices for this subset
    array[N] int tbl = bl[dstart:dend]; // Block index
    array[N] int tvalence = valence[dstart:dend]; // Valence for this subset
    
    array[N] int tchoice = choice[dstart:dend]; // Choice
    vector[N] toutcome = outcome[dstart:dend]; // Outcome

    real ttarget = 0.; // Log density for this subset

    // Correct participant indices
    for (i in 1:N){
        tpp[i] -= (start - 1);
    }

    // Loop over trials, updating EVs and incrementing log-density
    {    
      real PE;      // prediction error
      vector[2] ev; // expected value
  
      for (i in 1:N) {

        // Set ev to initial values times rho if participant/block different from previous trial
        if ((i == 1) || ((tpp[i] != tpp[i-1]) || (tbl[i] != tbl[i-1])))
          ev = rep_vector(aao * tvalence[i] * z[tpp[i]][1], 2);

        // Increment log densitiy
        ttarget += bernoulli_logit_lpmf(tchoice[i] | ev[2] - ev[1]);

        // prediction error
        PE = toutcome[i] * z[tpp[i]][1] - ev[tchoice[i] + 1];

        // value updating (learning)
        ev[tchoice[i] + 1] += z[tpp[i]][2] * PE;
      }
    }

    return ttarget;
  }
}

data {
  int<lower=1> N; // Total number of trials
  int<lower=1> N_p; // Number of participants
  int<lower=1> N_bl; // Number of blocks

  // Indices
  array[N] int<lower=1, upper=N_p> pp; // Participant number for each trial
  array[N_p+1] int<lower=1, upper=N+1> p1t; // First trial for each participant
  array[N] int<lower=1, upper=N_bl> bl; // Block number


  // Data
  array[N] int<lower=-1, upper=1> valence; // Valence of block - punishment or reward
  array[N] int<lower=0, upper=1> choice; // Binomial choice, coded as 1 for optimal stimulus, 0 for other stimulus 
  vector[N] outcome;  // Outcome observed, unbounded

  // Model constants
  real aao;  // average absolute outcome per block, for initial values for EV
  int grainsize; // For parrallelization
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Subject-level deviations
  vector[N_p] a;    // standardized learning rate deviations
  vector[N_p] rho;  // standardized reward sensitivity deviations
}

model {
  // Regularizing non-heirarchical prior
  a ~ normal(0, 1);
  rho ~ normal(0, 1);

  // Initiate subject-level parameters: first column is rho, second is alpha
  array[N_p] vector[2] z;
  
  // Compute subject-level parameters
  for (i in 1:N_p){
    z[i][1] = rho[i];
    z[i][2] = Phi_approx(a[i]);
  }
  
  target += reduce_sum(partial_sum, 
    z, 
    grainsize, 
    pp,
    p1t,
    bl,
    valence,
    choice,
    outcome,
    aao);
}

