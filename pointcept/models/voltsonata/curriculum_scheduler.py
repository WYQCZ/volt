import math
import logging

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    def __init__(
        self,
        warmup_ratio=0.2,
        curriculum_schedule="cosine",
        total_steps=0,
    ):
        self.warmup_ratio = warmup_ratio
        self.curriculum_schedule = curriculum_schedule
        self.total_steps = total_steps
        self.alpha = 0.0
        self.current_step = 0

    def before_train(self, total_steps, curr_step=0):
        self.total_steps = total_steps
        self.current_step = curr_step
        self.alpha = 0.0

    def step(self):
        if self.total_steps <= 0:
            return 0.0
        progress = self.current_step / self.total_steps
        if progress < self.warmup_ratio:
            alpha = 0.0
        else:
            t = (progress - self.warmup_ratio) / (1.0 - self.warmup_ratio)
            t = max(0.0, min(1.0, t))
            if self.curriculum_schedule == "cosine":
                alpha = 0.5 * (1 - math.cos(math.pi * t))
            else:
                alpha = t
        self.alpha = alpha
        self.current_step += 1
        return alpha

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "alpha": self.alpha,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.alpha = state_dict["alpha"]
        self.total_steps = state_dict["total_steps"]
