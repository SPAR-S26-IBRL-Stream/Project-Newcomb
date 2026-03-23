# Coin game infra-bayes toy
# =========================


# ---------- Environments ----------
  
from ibrl.environments.Coin_tossing_game_envs import MatchEnv, ReverseTailsEnv

#adds class MatchEnv: First environment : agent gets reward 1 if action matches observation, 0 if it doesn't
#and class ReverseTailsEnv: Second environment : If HEADS : agent gets reward if mismatch, 0 if match. If TAILS : agents always gets 0.5, regardless if match or not. 

# ---------- Policies ----------

def policy_action(policy_map, obs): # refers to the policy for choosing action on this observation
    # policy_map defines the chosen policy , it is a dictionary : {0:1, 1:1} for TT, HT: {0:0,1:1}, TH: {0:1,1:0}, HH: {0:0,1:0}
    return policy_map[obs]

# ---------- Utility / outcome ----------

def make_outcome(obs, reward):
    # outcome is a dictionary
    return {"obs": obs, "reward": reward}

def U_reward(outcome):
    # The utility is the reward the agent recieves
    return outcome["reward"]

# ---------- Gluing (f ⊕_h g) ----------

def const_utility(v): #Builds a utility function that ignores the outcome and returns either 0 or 1
    def u(_outcome):
        return float(v)
    return u

def event_obs_is(target_obs): #checks if an outcome is on a given branch : event_obs_is(0)
    def event(outcome):
        return outcome["obs"] == target_obs
    return event #boolean

def glue(f, event_h, g):  #example for 0 ⊕_h U : glue(const_utility(0.0), event_obs_is(0), U_reward) : if obs is HEADS, utility is 0, else is original reward
    # (f ⊕_h g)(outcome) = f(outcome) if h else g(outcome)
    def glued(outcome):
        if event_h(outcome):
            return f(outcome)
        return g(outcome)
    return glued

# ---------- Ordinary expectation in one environment ----------
# Computes the classical expected value for an environment and a policy, as in exercise 1

def expected_utility(env, policy_map, util_func):
    total = 0.0
    for obs in [0, 1]:
        p = 0.5
        action = policy_action(policy_map, obs)
        r = env.reward_table[obs][action]
        out = make_outcome(obs, r)
        total += p * util_func(out)
    return total

# ---------- Infra distribution H ----------

class InfraDist:
    # entries: list of dicts like {"env": env_obj, "lam": 1.0, "c": 0.0}
    def __init__(self, entries):
        self.entries = entries

    def tilde_E(self, policy_map, util_func): # computes the infra-expected value ~E_H[U], for a policy, across envs :
                                              #E_H[U] = min_e (lam * (E_e[U] + c))
        values = []
        for entry in self.entries:
            env = entry["env"]
            lam = entry.get("lam", 1.0) #lam short for lambda
            c = entry.get("c", 0.0)

            ev = expected_utility(env, policy_map, util_func)
            values.append(lam * (ev + c))

        return min(values)

    def tilde_P(self, policy_map, event_h, base_util): #computes infra -probability, for a policy across envs : ~P_H(h) = ~E_H[1⊕_h U] - ~E_H[0⊕_h U]
        # example, heads : computes worst case value if heads-branch is forces to 1, minus worst case value if h-branch forced to 0
        #difference tells how much "value weight" the branch heads has.
        u1 = glue(const_utility(1.0), event_h, base_util)
        u0 = glue(const_utility(0.0), event_h, base_util)
        return self.tilde_E(policy_map, u1) - self.tilde_E(policy_map, u0)

    def cond_value(self, policy_map, event_h, base_util):
        # Here is how the agent updates the infradistributions after making an observation, ie. after seeing HEADS or TAILS.
        # Updated infradistribution, E_H[U] becomes : ~E_{H|h}[U] = (~E(U)-~E(0⊕U)) / (~E(1⊕U)-~E(0⊕U))
        u1 = glue(const_utility(1.0), event_h, base_util)
        u0 = glue(const_utility(0.0), event_h, base_util)

        tE  = self.tilde_E(policy_map, base_util)
        tE0 = self.tilde_E(policy_map, u0)
        tE1 = self.tilde_E(policy_map, u1)

        num = tE - tE0
        den = tE1 - tE0

        if abs(den) < 1e-12:
            return 0.0  # ensure not divided by 0

        return num / den

def choose_best_policy_infra(H, policies):
    # Pick the policy with highest worst-case value (infra value)
    best_name = None
    best_pol = None
    best_val = None

    for name, pol in policies:
        val = H.tilde_E(pol, U_reward)
        if (best_val is None) or (val > best_val):
            best_val = val
            best_name = name
            best_pol = pol

    return best_name, best_pol, best_val


# ---------- MAIN ----------

def main():
    # Define the policies
    policies = [
        ("HH", {0: 0, 1: 0}),
        ("HT", {0: 0, 1: 1}),
        ("TH", {0: 1, 1: 0}),
        ("TT", {0: 1, 1: 1}),
    ]

    # Bucket H contains two environments with lambda=1, c=0
    H = InfraDist([
        {"env": MatchEnvironment(), "lam": 1.0, "c": 0.0},
        {"env": ReverseTailsEnvironment(), "lam": 1.0, "c": 0.0},
    ])

    print("Ex ante infra-values: ~E_H[U]")
    for name, pol in policies:
        print(name, H.tilde_E(pol, U_reward))

        # --- INFRA AGENT CONCLUSION (the decision) ---
    best_name, best_pol, best_val = choose_best_policy_infra(H, policies)

    print("\n CONCLUSION: ")
    print("Chosen policy (maximizes worst-case / infra value):", best_name, "value:", round(best_val, 2))
    print("So the agent will take the following action after observing:")

    for obs in [0, 1]:
        obs_name = "H" if obs == 0 else "T"
        action = policy_action(best_pol, obs)
        action_name = "H" if action == 0 else "T"
        print("  if observe", obs_name, "-> play", action_name)

    for obs in [0, 1]:
        event_h = event_obs_is(obs)
        obs_name = "H" if obs == 0 else "T"
        print("\n Update on observation:", obs_name)
        print("policy   tildeE(U)  tildeE(0⊕U)  tildeE(1⊕U)  tildeP(h)  updated")

        for name, pol in policies:
            tE  = H.tilde_E(pol, U_reward)
            tE0 = H.tilde_E(pol, glue(const_utility(0.0), event_h, U_reward))
            tE1 = H.tilde_E(pol, glue(const_utility(1.0), event_h, U_reward))
            tP  = tE1 - tE0
            upd = H.cond_value(pol, event_h, U_reward)

            # basic formatting
            print(
                name.ljust(5),
                str(round(tE,  2)).rjust(8),
                str(round(tE0, 2)).rjust(12),
                str(round(tE1, 2)).rjust(12),
                str(round(tP,  2)).rjust(9),
                str(round(upd, 2)).rjust(8),
            )

if __name__ == "__main__":
    main()
