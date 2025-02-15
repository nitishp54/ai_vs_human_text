"""Domain errors with stable codes for APIs and logs."""


class AppError(Exception):
    """Base application error."""

    code: str = "app_error"
    status_code: int = 500

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details or {}


class ConfigurationError(AppError):
    code = "configuration_error"
    status_code = 500


class ModelNotLoadedError(AppError):
    code = "model_not_loaded"
    status_code = 503


class PredictionError(AppError):
    code = "prediction_error"
    status_code = 422


class InputValidationError(AppError):
    code = "validation_error"
    status_code = 400


class LLMUnavailableError(AppError):
    code = "llm_unavailable"
    status_code = 503


class LLMProviderError(AppError):
    code = "llm_provider_error"
    status_code = 502


class ExternalServiceError(AppError):
    code = "external_service_error"
    status_code = 502
