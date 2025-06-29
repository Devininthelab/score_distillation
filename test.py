import torch

min_step = 0
max_step = 1000
t = torch.randint(
            self.min_step, self.max_step, (src_latents.shape[0],), 
            device=self.device, dtype=torch.long,
        )
t_prev = t - 1