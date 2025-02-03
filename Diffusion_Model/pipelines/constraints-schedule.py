class GradientConstraint():
    def __init__(self, beta_max=0.1, beta_min=1e-4, schedule='exponential'):
        super().__init__()
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.schedule = schedule

    def get_beta(self, t, max_t=999):
        # Convert t to progress (0->1)
        progress = 1 - t/max_t  # 0 at start, 1 at end

        if self.schedule == 'exponential':
            # Smooth exponential growth
            return self.beta_min + (self.beta_max - self.beta_min) * torch.exp(progress * 4 - 4)

        elif self.schedule == 'sigmoid':
            # Sigmoid transition
            x = 10 * (progress - 0.5)  # centered sigmoid
            beta = self.beta_min + (self.beta_max - self.beta_min) * (1 / (1 + torch.exp(-x)))
            return beta

        elif self.schedule == 'cosine':
            # Cosine schedule (smoother transition)
            cos_prog = 0.5 * (1 + torch.cos(torch.pi * (1 - progress)))
            return self.beta_min + (self.beta_max - self.beta_min) * (1 - cos_prog)

        elif self.schedule == 'power':
            # Power schedule (adjustable curve)
            power = 4  # higher = more sudden transition
            return self.beta_min + (self.beta_max - self.beta_min) * (progress ** power)

        elif self.schedule == 'delayed_exponential':
            # Stays very low until later in the process
            threshold = 0.7  # Start increasing significantly at 70% through
            if progress < threshold:
                scaled_progress = progress / threshold * 0.1  # Very slow increase
            else:
                scaled_progress = 0.1 + 0.9 * ((progress - threshold)/(1 - threshold))
            return self.beta_min + (self.beta_max - self.beta_min) * torch.exp(scaled_progress * 4 - 4)

        return self.beta_max  # default case


import matplotlib.pyplot as plt
import numpy as np

def plot_schedules():
    t = np.linspace(999, 0, 1000)
    constraint = GradientConstraint(beta_max=1.0, beta_min=0.01)

    schedules = ['exponential', 'sigmoid', 'cosine', 'power', 'delayed_exponential']
    plt.figure(figsize=(12, 6))

    for schedule in schedules:
        constraint.schedule = schedule
        betas = [constraint.get_beta(torch.tensor(ti)).item() for ti in t]
        plt.plot(range(1000), betas, label=schedule)

    plt.xlabel('Diffusion Step')
    plt.ylabel('Beta Value')
    plt.legend()
    plt.title('Different Beta Schedules')
    plt.grid(True)
    plt.show()
