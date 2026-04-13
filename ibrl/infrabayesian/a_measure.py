import numpy as np


class AMeasure:
    """
    An a-measure, characterised by
        a scale factor λ > 0
        a probability measure μ
        an offset b > 0

    The probability measure assigns a probability to all possible histories.
    This could be implemented by writing out all histories explicitly (up to some depth) and computing the
    probabilities of each. This is computationally infeasible, so instead we parameterise the measure. We can construct
    measure components for arbitrary histories in constant time (per history).

    Start by considering a pure measure (one that is not a mixture):

    The probability of outcome i occurring is p[i].
    Say we are at some point in history, where each outcome i has occurred o[i] times.
    The probability of arriving at this point is
        (Σ_i p[i]^o[i])
    The probability of the next outcome being k is
        (Σ_i p[i]^o[i]) p[k]
    We divide by the probability of reaching this point to get the probability distribution over the next outcome:
        (Σ_i p[i]^o[i]) p[k] / (Σ_i p[i]^o[i]) = p[k]
    This simplification is expected and trivial for a pure measure.

    Now consider a mixed measure, i.e. one that is a linear combination of pure measures:
    We mix several pure measures with coefficients c[j].
    The pure measure j assigns probability p[j,i] to outcome i.
    Analogously to above, the probability of reaching a given point in history is
        Σ_j c[j] Σ_i p[j,i]^o[i]
    The probability of the next outcome being k is
        Σ_j c[j] Σ_i p[j,i]^o[i] p[j,k]
    And thus the probability distribution over the next outcome is
        (Σ_j c[j] Σ_i p[j,i]^o[i] p[j,k]) / (Σ_j c[j] Σ_i p[j,i]^o[i])
    
    We no longer get the cancellation from the pure measure case, but the computational complexity is still
        O( #(pure measures) * #(outcomes) )
    rather than
        O( 2^(history depth) )
    In particular, we do not need to fix the history depth at the beginning.

    Arguments:
        probabilities:  2D array of p[j,i]
        coefficients:   1D array of c[j]
    """
    def __init__(self, probabilities : np.ndarray, coefficients : np.ndarray, scale : float = 1, offset : float = 0):
        self.num_components = probabilities.shape[0]
        self.num_outcomes = probabilities.shape[1]
        assert probabilities.shape == (self.num_components,self.num_outcomes)
        assert coefficients.shape  == (self.num_components,)
        assert offset >= 0
        assert scale > 0
        self.log_probabilities = np.log(np.maximum(probabilities,1e-300))  # avoid log(0)
        self.coefficients = coefficients
        self.scale = scale
        self.offset = offset

    def compute_probabilities(self, history : np.ndarray) -> np.ndarray:
        """
        Compute probabilities of next outcomes, given the previously observed history

        history is a (num_outcomes,) integer array, which indicates how often each outcome has been observed

        Return a (num_outcomes,) array of probabilities
        """
        # Log prob of current history, under each component. Shape (num_components,1)
        log_probs = np.expand_dims((self.log_probabilities*history).sum(axis=1),axis=1)
        # Probability of immediately following histories, summed over components. Shape (num_outcomes,)
        probs = self.coefficients @ np.exp(log_probs + self.log_probabilities)
        # Normalise (divide by probability of reaching this point in history)
        return probs / probs.sum()

    def expected_value(self, history : np.ndarray, reward_function : np.ndarray) -> float:
        """
        Compute the expected value of a given reward function, defined as λ*μ(f) + b
        """
        return self.scale * (self.compute_probabilities(history) @ reward_function) + self.offset

    # Helper functions

    def __itruediv__(self, other : float):
        """
        In-place division by a scalar
        """
        self.scale /= other
        self.offset /= other
        return self

    def __repr__(self) -> str:
        """
        String representation: "(λμ,b)"
        where μ is a list of all components "[component1,component2,...]"
        and each component is "c:{p1,p2,...}" where c is the mixing coefficient and pi is the probability of outcome i
        """
        components = []
        for c,p in zip(self.coefficients,np.exp(self.log_probabilities)):
            components.append(f"""{c:.2f}:{{{",".join(f"{pp:.1f}" for pp in p)}}}""")
        return f"""({self.scale:.3f}[{",".join(components)}],{self.offset:.3f})"""
