# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain_community.chat_models import ChatOllama
from pydantic import Field
from typing import Optional
from chatui.utils import gpu_compatibility


class CustomChatOpenAI(ChatOllama):
    """Thin wrapper around ChatOllama to match previous NIM interface."""

    custom_endpoint: str = Field(None, description="Endpoint of remote Ollama server")
    port: Optional[str] = "11434"
    model_name: Optional[str] = "llama3.1:8b-instruct-q8_0"
    temperature: Optional[float] = 0.0
    gpu_type: Optional[str] = None
    gpu_count: Optional[str] = None

    def __init__(
        self,
        custom_endpoint,
        port: str = "11434",
        model_name: str = "llama3.1:8b-instruct-q8_0",
        gpu_type: Optional[str] = None,
        gpu_count: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ):
        if gpu_type and gpu_count:
            compatibility = gpu_compatibility.get_compatible_models(gpu_type, gpu_count)
            if compatibility["warning_message"]:
                raise ValueError(compatibility["warning_message"])
            if model_name not in compatibility["compatible_models"]:
                raise ValueError(
                    f"Model {model_name} is not compatible with {gpu_type} ({gpu_count} GPUs)"
                )
        super().__init__(
            base_url=f"http://{custom_endpoint}:{port}",
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
        self.custom_endpoint = custom_endpoint
        self.port = port
        self.model_name = model_name
        self.temperature = temperature
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
