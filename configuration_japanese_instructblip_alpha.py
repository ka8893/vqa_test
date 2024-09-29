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
""" Japanese InstructBLIP Alpha model configuration"""

from transformers import (
    PretrainedConfig,
    InstructBlipConfig, 
    InstructBlipVisionConfig,
    InstructBlipQFormerConfig,
    AutoConfig,
)
from transformers.utils import logging
from .configuration_japanese_stablelm_alpha import JapaneseStableLMAlphaConfig


logger = logging.get_logger(__name__)


class JapaneseInstructBlipAlphaConfig(InstructBlipConfig):
    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        PretrainedConfig.__init__(self, **kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the InstructBlipVisionConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the InstructBlipQFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")
        self.vision_config = InstructBlipVisionConfig(**vision_config)
        self.qformer_config = InstructBlipQFormerConfig(**qformer_config)
        self.text_config = JapaneseStableLMAlphaConfig(**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = True
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
