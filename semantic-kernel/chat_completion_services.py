# Copyright (c) Microsoft. All rights reserved.

from enum import Enum
from typing import TYPE_CHECKING

from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings


class Services(str, Enum):
    """Enum for supported chat completion services.

    For service specific settings, refer to this documentation:
    https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/setup/ALL_SETTINGS.md
    """

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    AZURE_AI_INFERENCE = "azure_ai_inference"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    GOOGLE_AI = "google_ai"
    MISTRAL_AI = "mistral_ai"
    OLLAMA = "ollama"
    ONNX = "onnx"
    VERTEX_AI = "vertex_ai"
    DEEPSEEK = "deepseek"


service_id = "default"


def get_chat_completion_service_and_request_settings(
    service_name: Services,
    instruction_role: str | None = None,
) -> tuple["ChatCompletionClientBase", "PromptExecutionSettings"]:
    """Return service and request settings.

    Args:
        service_name (Services): The service name.
        instruction_role (str | None): The role to use for 'instruction' messages, for example,
            'system' or 'developer'. Defaults to 'system'. Currently only OpenAI reasoning models
            support 'developer' role.
    """
    # Use lambdas or functions to delay instantiation
    chat_services = {
        Services.OPENAI: lambda: get_openai_chat_completion_service_and_request_settings(
            instruction_role=instruction_role
        ),
        Services.AZURE_OPENAI: lambda: get_azure_openai_chat_completion_service_and_request_settings(
            instruction_role=instruction_role
        ),
        Services.AZURE_AI_INFERENCE: lambda: get_azure_ai_inference_chat_completion_service_and_request_settings(
            instruction_role=instruction_role
        ),
        Services.ANTHROPIC: lambda: get_anthropic_chat_completion_service_and_request_settings(),
        Services.BEDROCK: lambda: get_bedrock_chat_completion_service_and_request_settings(),
        Services.GOOGLE_AI: lambda: get_google_ai_chat_completion_service_and_request_settings(),
        Services.MISTRAL_AI: lambda: get_mistral_ai_chat_completion_service_and_request_settings(),
        Services.OLLAMA: lambda: get_ollama_chat_completion_service_and_request_settings(),
        Services.ONNX: lambda: get_onnx_chat_completion_service_and_request_settings(),
        Services.VERTEX_AI: lambda: get_vertex_ai_chat_completion_service_and_request_settings(),
        Services.DEEPSEEK: lambda: get_deepseek_chat_completion_service_and_request_settings(),
    }

    # Call the appropriate lambda or function based on the service name
    if service_name not in chat_services:
        raise ValueError(f"Unsupported service name: {service_name}")
    return chat_services[service_name]()


def get_openai_chat_completion_service_and_request_settings(
    instruction_role: str | None = None,
) -> tuple["ChatCompletionClientBase", "PromptExecutionSettings"]:
    """Return OpenAI chat completion service and request settings.

    Args:
        instruction_role (str | None): The role to use for 'instruction' messages, for example,
            'developer' or 'system'. (Optional)

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel-python
    """
    from semantic_kernel.connectors.ai.open_ai import OpenAISettings
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings

    # Load settings from .env file
    openai_settings = OpenAISettings.create(env_file_path=".env")

    chat_service = OpenAIChatCompletion(
        service_id=service_id, 
        ai_model_id=openai_settings.chat_model_id,
        instruction_role=instruction_role,
        env_file_path=".env"
    )
    
    request_settings = OpenAIChatPromptExecutionSettings(
        service_id=service_id, 
        max_tokens=2000, 
        temperature=0.7, 
        top_p=0.8
    )

    return chat_service, request_settings


def get_azure_openai_chat_completion_service_and_request_settings(
    instruction_role: str | None = None,
) -> tuple["ChatCompletionClientBase", "PromptExecutionSettings"]:
    """Return Azure OpenAI chat completion service and request settings.

    Args:
        instruction_role (str | None): The role to use for 'instruction' messages, for example,
            'developer' or 'system'. (Optional)

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
    from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings
    # Load settings from .env file
    azure_settings = AzureOpenAISettings.create(env_file_path=".env")

    chat_service = AzureChatCompletion(
        service_id=service_id, 
        deployment_name=azure_settings.chat_deployment_name,
        endpoint=azure_settings.endpoint,
        api_key=azure_settings.api_key.get_secret_value() if azure_settings.api_key else None,
        api_version=azure_settings.api_version,
        instruction_role=instruction_role,
        env_file_path=".env"
    )
    
    request_settings = AzureChatPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings


def get_azure_ai_inference_chat_completion_service_and_request_settings(
    instruction_role: str | None = None,
) -> tuple["ChatCompletionClientBase", "PromptExecutionSettings"]:
    """Return Azure AI Inference chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.azure_ai_inference import (
        AzureAIInferenceChatCompletion,
        AzureAIInferenceChatPromptExecutionSettings,
    )

    # The AI model ID is used as an identifier for developers when they are using serverless endpoints
    # on AI Foundry. It is not actually used to identify the model in the service as the endpoint points
    # to only one model.
    # When developers are using one endpoint that can route to multiple models, the `ai_model_id` will be
    # used to identify the model. To use the latest routing feature on AI Foundry, please refer to the
    # following documentation:
    # https://learn.microsoft.com/en-us/azure/ai-services/multi-service-resource?%3Fcontext=%2Fazure%2Fai-services%2Fmodel-inference%2Fcontext%2Fcontext&pivots=azportal
    # https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/how-to/configure-project-connection?pivots=ai-foundry-portal
    # https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/how-to/inference?tabs=python

    chat_service = AzureAIInferenceChatCompletion(
        service_id=service_id,
        ai_model_id="id",
        instruction_role=instruction_role,
    )
    request_settings = AzureAIInferenceChatPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings


def get_anthropic_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return Anthropic chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion, AnthropicChatPromptExecutionSettings

    chat_service = AnthropicChatCompletion(service_id=service_id)
    request_settings = AnthropicChatPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings


def get_bedrock_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return Bedrock chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.bedrock import BedrockChatCompletion, BedrockChatPromptExecutionSettings

    chat_service = BedrockChatCompletion(service_id=service_id, model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    request_settings = BedrockChatPromptExecutionSettings(
        # For model specific settings, specify them in the extension_data dictionary.
        # For example, for Cohere Command specific settings, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        service_id=service_id,
        extension_data={
            "temperature": 0.8,
        },
    )

    return chat_service, request_settings


def get_google_ai_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return Google AI chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.google.google_ai import (
        GoogleAIChatCompletion,
        GoogleAIChatPromptExecutionSettings,
    )

    chat_service = GoogleAIChatCompletion(service_id=service_id)
    request_settings = GoogleAIChatPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings


def get_mistral_ai_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return Mistral AI chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.mistral_ai import MistralAIChatCompletion, MistralAIChatPromptExecutionSettings

    chat_service = MistralAIChatCompletion(service_id=service_id)
    request_settings = MistralAIChatPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings


def get_ollama_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return Ollama chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
    from semantic_kernel.connectors.ai.ollama.ollama_settings import OllamaSettings
    
    # Load settings from .env file
    ollama_settings = OllamaSettings.create(env_file_path=".env")
    if not ollama_settings.chat_model_id:
        # Default to llama3 if no model specified
        ollama_settings.chat_model_id = "llama3"
    
    chat_service = OllamaChatCompletion(
        service_id=service_id, 
        ai_model_id=ollama_settings.chat_model_id,
        host=ollama_settings.host,
        env_file_path=".env"
    )
    
    request_settings = OllamaChatPromptExecutionSettings(
        # For model specific settings, specify them in the options dictionary.
        # For more information on the available options, refer to the Ollama API documentation:
        # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        service_id=service_id,
        options={
            "temperature": 0.8,
        },
    )

    return chat_service, request_settings


def get_onnx_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return Onnx chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.onnx import (
        OnnxGenAIChatCompletion,
        OnnxGenAIPromptExecutionSettings,
        ONNXTemplate,
    )

    chat_service = OnnxGenAIChatCompletion(ONNXTemplate.PHI3, service_id=service_id)
    request_settings = OnnxGenAIPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings


def get_vertex_ai_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return Vertex AI chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel
    """
    from semantic_kernel.connectors.ai.google.vertex_ai import (
        VertexAIChatCompletion,
        VertexAIChatPromptExecutionSettings,
    )

    chat_service = VertexAIChatCompletion(service_id=service_id)
    request_settings = VertexAIChatPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings


def get_deepseek_chat_completion_service_and_request_settings() -> tuple[
    "ChatCompletionClientBase", "PromptExecutionSettings"
]:
    """Return DeepSeek chat completion service and request settings.

    The service credentials can be read by 3 ways:
    1. Via the constructor
    2. Via the environment variables
    3. Via an environment file

    The DeepSeek endpoint can be accessed via the OpenAI connector as the DeepSeek API is compatible with OpenAI API.
    Set the `DEEPSEEK_API_KEY` environment variable to the DeepSeek API key.
    Set the `DEEPSEEK_MODEL_ID` environment variable to the DeepSeek model ID (deepseek-chat or deepseek-reasoner).

    The request settings control the behavior of the service. The default settings are sufficient to get started.
    However, you can adjust the settings to suit your needs.
    Note: Some of the settings are NOT meant to be set by the user.
    Please refer to the Semantic Kernel Python documentation for more information:
    https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel?view=semantic-kernel-python
    """
    from openai import AsyncOpenAI
    import os
    from semantic_kernel.connectors.ai.open_ai import (
        OpenAIChatCompletion,
        OpenAIChatPromptExecutionSettings,
        OpenAISettings,
    )

    # Load settings from .env file
    openai_settings = OpenAISettings.create(env_file_path=".env")
    
    # Check for DeepSeek specific environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY", openai_settings.api_key.get_secret_value() if openai_settings.api_key else None)
    model_id = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
    
    if not api_key:
        raise ServiceInitializationError("The DeepSeek API key is required.")
    
    chat_service = OpenAIChatCompletion(
        ai_model_id=model_id,
        service_id=service_id,
        async_client=AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        ),
        env_file_path=".env"
    )
    request_settings = OpenAIChatPromptExecutionSettings(service_id=service_id)

    return chat_service, request_settings