import jax.numpy as jnp
import jax.scipy as jscipy
import numpyro
import numpyro.distributions as npdist
from jax import lax


def sample_weight_priors(condition, num_features, weight_mean, weight_scale, t=-1):
    if condition == "none":
        # no additional information about the weights
        weights = numpyro.sample(
            f"weights_{t}", npdist.Normal(weight_mean, weight_scale)
        )

    elif condition == "direction":
        # Force positive weights using absolute value
        weights = numpyro.sample(
            f"weights_{t}",
            npdist.HalfNormal(weight_scale),
        )

    elif condition == "equal_weighing":
        # Heuristic where all weights have the same weight
        # Sample a single weight value that will be used for all features
        shared_weight = numpyro.sample(
            f"shared_weight_{t}",
            npdist.Normal(weight_mean[0], weight_scale[0]),
        )
        # Create a vector of identical weights
        weights = numpyro.deterministic(
            f"weights_{t}", jnp.ones(num_features) * shared_weight
        )
    elif condition == "ranking_sort":
        # abs value of the weights is sorted by magnitude (first has lowest abs
        # value and last has highest)
        # Sample raw weights from normal distribution
        raw_weights = numpyro.sample(
            f"raw_weights_{t}", npdist.Normal(weight_mean, weight_scale)
        )

        sorted_magnitudes = jnp.sort(jnp.abs(raw_weights), axis=1)
        direction = jnp.tanh(raw_weights * 2.0)
        # Combine magnitudes with smooth direction
        weights = numpyro.deterministic(f"weights_{t}", sorted_magnitudes * direction)
    elif condition == "single_cue":
        # Heuristic where only last feature has non-zero weight
        # Sample a single weight for the last feature
        last_weight = numpyro.sample(
            f"last_weight_{t}",
            npdist.Normal(weight_mean[-1], weight_scale[-1]),
        )
        # Create a vector of zeros with the last element set to the sampled weight
        weights = numpyro.deterministic(
            f"weights_{t}",
            jnp.where(jnp.arange(num_features) == num_features - 1, last_weight, 0.0),
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return weights


def ideal_observer_model(
    inputs,  # shape = (num_experiments, num_trials, num_features)
    targets,  # shape = (num_experiments, num_trials)
    answer_probs,  # shape = (num_experiments, num_trials)
    condition,
):
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    answer_probs = jnp.array(answer_probs)

    num_experiments = inputs.shape[0]
    num_trials = inputs.shape[1]
    num_features = inputs.shape[2]

    sigma = numpyro.sample("sigma", npdist.HalfNormal(0.1))
    weight_mean = jnp.zeros(num_features)
    weight_scale = numpyro.sample("weight_scale", npdist.HalfNormal(1.0))
    weight_scale = jnp.ones(num_features) * weight_scale
    prediction_scale = numpyro.sample("prediction_scale", npdist.HalfNormal(0.1))

    with numpyro.plate("experiments", num_experiments, dim=-2):
        weights = sample_weight_priors(
            condition, num_features, weight_mean, weight_scale
        )

        with numpyro.plate("trials", num_trials, dim=-1):
            # Compute decision variable Y = w^T x for each experiment and trial
            Y = numpyro.deterministic(
                "Y", jnp.sum(weights[..., None, :] * inputs, axis=-1)
            )

            # Probability of choosing option 1
            probs = numpyro.deterministic(
                "probs", jscipy.special.ndtr(Y / (jnp.sqrt(2.0) * sigma + 1e-6))
            )

            numpyro.sample("targets", npdist.Bernoulli(probs), obs=targets)

            numpyro.sample(
                "answer_probs",
                npdist.TruncatedNormal(probs, prediction_scale, low=0.0, high=1.0),
                obs=answer_probs,
            )
