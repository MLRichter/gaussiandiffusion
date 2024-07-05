import torch


def stratified_random_sampling(num_samples):
    # Divide the interval [0, 1] into `num_samples` strata
    strata_edges = torch.linspace(0, 1, num_samples + 1)

    # Sample one random point within each stratum and perturb it
    samples = []
    for i in range(num_samples):
        start, end = strata_edges[i], strata_edges[i + 1]
        sample = torch.distributions.Uniform(start, end).sample()
        samples.append(sample)

    return torch.tensor(samples)


# Example usage
num_samples = 10
samples = stratified_random_sampling(num_samples)
print(samples)