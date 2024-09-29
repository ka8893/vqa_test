# coding=utf-8
# Copyright 2023 Stability and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch JapaneseStableLMAlpha model. """
import torch
from torch import nn
from transformers import (
    InstructBlipPreTrainedModel, 
    InstructBlipVisionModel, 
    InstructBlipQFormerModel,
    InstructBlipForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers.utils import logging
from .modeling_japanese_stablelm_alpha import JapaneseStableLMAlphaForCausalLM
from .configuration_japanese_instructblip_alpha import JapaneseInstructBlipAlphaConfig


logger = logging.get_logger(__name__)


class JapaneseInstructBlipAlphaForConditionalGeneration(InstructBlipForConditionalGeneration):
    config_class = JapaneseInstructBlipAlphaConfig

    def __init__(self, config: JapaneseInstructBlipAlphaConfig):
        InstructBlipPreTrainedModel.__init__(self, config)

        self.vision_model = InstructBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        if config.use_decoder_only_language_model:
            language_model = JapaneseStableLMAlphaForCausalLM(config.text_config)
        else:
            raise NotImplementedError
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config, trust_remote_code=True,)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()
