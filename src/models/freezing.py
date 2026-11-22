class EncoderDecoderFreezingMixing:
    """Control gradients and training mode independently for both model parts."""

    def set_freezing(self, freeze_encoder=False, freeze_decoder=False):
        if not isinstance(freeze_encoder, bool) or not isinstance(freeze_decoder, bool):
            raise TypeError("freeze_encoder and freeze_decoder must be YAML booleans")
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.encoder.requires_grad_(not freeze_encoder)
        self.decoder.requires_grad_(not freeze_decoder)
        # Clear stale gradients when freezing an already-used model.
        for module, frozen in ((self.encoder, freeze_encoder), (self.decoder, freeze_decoder)):
            if frozen:
                for parameter in module.parameters():
                    parameter.grad = None
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm statistics and dropout must also stay fixed.
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
        if getattr(self, "freeze_decoder", False):
            self.decoder.eval()
        return self
