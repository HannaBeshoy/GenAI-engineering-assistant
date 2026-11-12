"""Errors the app shows to the user. Every failure ends in one of these, with a message
that says what went wrong and what to do, instead of a stack trace."""


class AssistantError(Exception):
    """Base class: `user_message` is what the Streamlit page shows."""

    hint = ""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint or self.hint

    @property
    def user_message(self) -> str:
        return f"{self}" + (f" {self.hint}" if self.hint else "")


class ConfigError(AssistantError):
    hint = "Check the provider, model and API key in the sidebar (or in the .env file)."


class AuthError(AssistantError):
    hint = "The API key was rejected: paste a valid key in the sidebar."


class RateLimitError(AssistantError):
    hint = ("The provider's rate limit was hit even after waiting and retrying. Wait a minute and try again, or "
            "lower max_tokens in the sidebar (free tiers such as Groq's count max_tokens toward the tokens-per-minute "
            "limit).")


class LLMTimeout(AssistantError):
    hint = "The provider did not answer in time. Try again, or lower max_tokens."


class ProviderUnavailable(AssistantError):
    hint = "The provider could not be reached (for Ollama: is `ollama serve` running?)."


class TruncatedOutput(AssistantError):
    hint = "The answer was cut off by max_tokens. Raise max_tokens in the sidebar."


class BadModelOutput(AssistantError):
    hint = "The model kept returning output that does not match the expected format, even after repair attempts."


class PdfError(AssistantError):
    hint = "Upload a text-based PDF (not a scan or a password-protected file)."


class InputDataError(AssistantError):
    hint = "Check the uploaded file against the sample file's format."


class DemoModeError(AssistantError):
    hint = "Demo mode replays recorded answers for the sample files only. Choose a real provider and API key for your own files."
