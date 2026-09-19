from __future__ import annotations


class MIError(RuntimeError):
    """Expected user-facing Mixed Inference error."""

    def __init__(self, message: str, *, code: str = "mi_error", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


class ConfigurationError(MIError):
    def __init__(self, message: str):
        super().__init__(message, code="invalid_configuration", status=400)


class CompatibilityError(MIError):
    def __init__(self, message: str):
        super().__init__(message, code="incompatible_artifact", status=409)


class CompositionError(MIError):
    def __init__(self, message: str):
        super().__init__(message, code="invalid_composition", status=409)


class MissingDependencyError(MIError):
    def __init__(self, message: str):
        super().__init__(message, code="missing_dependency", status=404)


class TrainingCancelled(MIError):
    def __init__(self, message: str = "Training was cancelled"):
        super().__init__(message, code="training_cancelled", status=409)
