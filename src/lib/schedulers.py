"""
Implementation of learning rate schedulers, early stopping and other utils
for improving optimization
"""

from src.lib.logger import print_


class ExponentialLRSchedule:
    """
    Exponential LR Scheduler that decreases the learning rate by multiplying it
    by an exponentially decreasing decay factor:
        LR = LR * gamma ^ (step/total_steps)

    Args:
    -----
    optimizer: torch.optim
        Optimizer to schedule
    init_lr: float
        base learning rate to decrease with the exponential scheduler
    gamma: float
        exponential decay factor
    total_steps: int/float
        number of optimization steps to optimize for. Once this is reached,
        lr is not decreased anymore
    """

    def __init__(self, optimizer, init_lr, gamma=0.5, total_steps=100_000):
        """ Module initializer """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.total_steps = total_steps
        return

    def update_lr(self, step):
        """ Computing exponential lr update """
        new_lr = self.init_lr * self.gamma ** (step / self.total_steps)
        return new_lr

    def step(self, iter):
        """ Scheduler step """
        if iter < self.total_steps:
            for params in self.optimizer.param_groups:
                params["lr"] = self.update_lr(iter)
        elif iter == self.total_steps:
            print_(f"Finished exponential decay due to reach of {self.total_steps} steps")
        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        return state_dict

    def load_state_dict(self, state_dict):
        """ Loading state dictinary """
        self.init_lr = state_dict["init_lr"]
        self.gamma = state_dict["gamma"]
        self.total_steps = state_dict["total_steps"]
        return


class LRWarmUp:
    """
    Class for performing learning rate warm-ups. We increase the learning rate
    during the first few iterations until it reaches the standard LR

    Args:
    -----
    init_lr: float
        initial learning rate
    warmup_steps: integer
        number of optimization steps to warm up for
    max_epochs: integer
        maximum number of epochs to warmup. It overrides 'warmup_step'
    """

    def __init__(self, init_lr, warmup_steps, max_epochs=1):
        """ Initializer """
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.active = True

    def __call__(self, iter, epoch, optimizer):
        if not self.active:
            return
        if self.warmup_steps <= 0 or iter >= self.warmup_steps or epoch >= self.max_epochs:
            self.active = False
            lr = self.init_lr
            print_("Finished learning rate warmup period...")
        else:
            lr = self.init_lr * (iter / self.warmup_steps)
        for parameters in optimizer.param_groups:
            parameters["lr"] = lr

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items()}
        return state_dict


class EarlyStop:
    """
    Implementation of an early stop criterion

    Args:
    -----
    mode: string ['min', 'max']
        whether we validate based on maximizing or minmizing a metric
    delta: float
        threshold to consider improvements
    use_early_stop: bool
        If True, early stopping functionalities are computed.
    patience: integer
        number of epochs without improvement to trigger early stopping
    """

    def __init__(self, mode="min", delta=1e-6, use_early_stop=True, patience=10):
        """ Early stopper initializer """
        assert mode in ["min", "max"]
        self.mode = mode
        self.delta = delta
        self.use_early_stop = use_early_stop
        self.patience = patience
        self.counter = 0

        if mode == "min":
            self.best = 1e15
            self.criterion = lambda x: x < (self.best - self.delta)
        elif mode == "max":
            self.best = 1e-15
            self.criterion = lambda x: x > (self.best + self.delta)

        return

    def __call__(self, value, epoch=0, writer=None):
        """
        Comparing current metric agains best past results and computing if we
        should early stop or not

        Args:
        -----
        value: float
            validation metric measured by the early stopping criterion
        wirter: TensorboardWriter
            If not None, TensorboardWriter to log the early-stopper counter

        Returns:
        --------
        stop_training: boolean
            If True, we should early stop. Otherwise, metric is still improving
        """
        if not self.use_early_stop:
            return False

        are_we_better = self.criterion(value)
        if are_we_better:
            self.counter = 0
            self.best = value
        else:
            self.counter = self.counter + 1

        stop_training = True if(self.counter >= self.patience) else False
        if stop_training:
            print_(f"It has been {self.patience} epochs without improvement")
            print_("  --> Trigering early stopping")

        if writer is not None:
            writer.add_scalar(
                name="Learning/Early-Stopping Counter",
                val=self.counter,
                step=epoch + 1,
            )

        return stop_training


class Freezer:
    """Manage encoder/decoder gradients and the modes of frozen modules.

    ``frozen_epochs=None`` applies the two freeze flags throughout training.
    An integer overrides the encoder flag: freeze for epochs [0, frozen_epochs),
    then unfreeze. Decoder freezing remains controlled independently.
    The original classifier-model constructor arguments remain supported.
    """

    def __init__(self, module, model_name=None, use_freezer=True, frozen_epochs=None,
                 *, freeze_encoder=False, freeze_decoder=False):
        if frozen_epochs is not None and (isinstance(frozen_epochs, bool)
                                          or not isinstance(frozen_epochs, int)
                                          or frozen_epochs < 0):
            raise ValueError("frozen_epochs must be a nonnegative integer or null")
        self.module = module
        self.model_name = model_name
        self.use_freezer = use_freezer
        self.frozen_epochs = frozen_epochs
        if hasattr(module, "encoder") and hasattr(module, "decoder"):
            self.encoder = module.encoder
            self.decoder = module.decoder
        else:
            # Compatibility with the classification models used by older callers.
            self.encoder = module
            if "ResNet" in (model_name or "") or model_name == "ResNext":
                self.decoder = module.fc
            elif model_name == "EfficientNet" or "VGG" in (model_name or ""):
                self.decoder = module.classifier[-1]
            elif "ConvNext" in (model_name or ""):
                self.decoder = module.head
            else:
                raise ValueError("Freezer requires encoder/decoder modules or a supported model_name")
        self.set_freezing(freeze_encoder, freeze_decoder)
        self(0)

    @property
    def is_frozen(self):
        return self.freeze_encoder

    def set_freezing(self, freeze_encoder=False, freeze_decoder=False):
        if not isinstance(freeze_encoder, bool) or not isinstance(freeze_decoder, bool):
            raise TypeError("freeze_encoder and freeze_decoder must be YAML booleans")
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.module.freeze_encoder = freeze_encoder
        self.module.freeze_decoder = freeze_decoder
        self.encoder.requires_grad_(not freeze_encoder)
        self.decoder.requires_grad_(not freeze_decoder)
        for part, frozen in ((self.encoder, freeze_encoder), (self.decoder, freeze_decoder)):
            if frozen:
                for parameter in part.parameters():
                    parameter.grad = None
        self.module.train(self.module.training)
        self.enforce_training_mode()
        return self

    def enforce_training_mode(self):
        training = self.module.training
        if self.freeze_encoder:
            self.encoder.eval()
            if self.encoder is self.module:
                # Older classifiers use the whole model as their encoder.
                # Retain its requested mode while freezing the feature layers.
                self.module.training = training
        self.decoder.train(training and not self.freeze_decoder)

    def __call__(self, epoch):
        if self.use_freezer and self.frozen_epochs is not None:
            frozen = epoch < self.frozen_epochs
            if frozen != self.freeze_encoder:
                self.set_freezing(frozen, self.freeze_decoder)
        self.enforce_training_mode()
