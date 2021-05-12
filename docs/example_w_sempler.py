import causalicp as icp
import sempler
import sempler.generators

# Generate a random graph and construct a linear-Gaussian SCM
W = sempler.generators.dag_avg_deg(4, 2.5, 0.5, 2)
scm = sempler.LGANM(W, (-1, 1), (1, 2))

# Generate a sample for setting 1: Observational setting
data = [scm.sample(n=100)]

# Setting 2: Shift-intervention on X1
data += [scm.sample(n=130, shift_interventions={1: (3.1, 5.4)})]

# Setting 3: Do-intervention on X2
data += [scm.sample(n=98, do_interventions={2: (-1, 3)})]


result = icp.fit(data, 3, alpha=0.05, precompute=True, verbose=True, color=False)
