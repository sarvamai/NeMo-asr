# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
import torch
from lhotse import MonoCut
from lhotse.cut import Cut, MixedCut
from lhotse.utils import ifnone
import numpy as np

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.canary import BOOL_FALSE, BOOL_TRUE, PNC_FALSE, PNC_TRUE
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers.canary_tokenizer import (
    CANARY2_BOCTX,
    CANARY_BOS,
    CANARY_EOS,
    CANARY_SPECIAL_TOKENIZER,
    CanaryTokenizer,
)
from nemo.collections.common.tokenizers.canary_multilingual_tokenizer import (
    CanaryMultilingualTokenizer,
)

import random
from collections import deque

# Use global variables to import slot values in other modules.
ITN_TRUE = BOOL_TRUE | {
    "itn",
    "<|itn|>",
}
ITN_FALSE = BOOL_FALSE | {"noitn", "<|noitn|>"}
TIMESTAMP_TRUE = BOOL_TRUE | {"timestamp", "<|timestamp|>"}
TIMESTAMP_FALSE = BOOL_FALSE | {"notimestamp", "<|notimestamp|>"}
DIARIZE_TRUE = BOOL_TRUE | {"diarize", "<|diarize|>"}
DIARIZE_FALSE = BOOL_FALSE | {"nodiarize", "<|nodiarize|>"}


# Global deque to store random contexts, with a maximum size.
RANDOM_CONTEXTS = deque(maxlen=1000)


class Canary2PromptFormatter(PromptFormatter):
    NAME = "canary2"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        # User prompt.
        "user": {
            "template": f"{CANARY2_BOCTX}|decodercontext|{CANARY_BOS}|emotion||pnc||itn||diarize||target_lang|",
            "slots": {
                # Empty string or previous transcript / other context to bias predictions.
                "decodercontext": Modality.Text,
                "emotion": Modality.Text,
                "pnc": Modality.Text,
                "itn": Modality.Text,
                "diarize": Modality.Text,
                "target_lang": Modality.Text,
            },
        },
        # System's reponse.
        OUTPUT_ROLE: {
            "template": f"|source_lang||text|{CANARY_EOS}",
            "slots": {
                "source_lang": Modality.Text,
                "text": Modality.Text,
            },
        },
    }

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        if "text" in expected_slots and "source_lang" in expected_slots:  # is assistant turn
            tokens = []
            # Split template by |text|
            before_text, after_text = prompt_template.split('|text|')

            # Process and tokenize before_text part
            prompt = before_text
            for slot in expected_slots:
                if slot != 'text':
                    prompt = prompt.replace(f'|{slot}|', slot_values[slot])
            tokens.extend(self._apply_tokenizer(prompt, lang=CANARY_SPECIAL_TOKENIZER))

            # Tokenize text part
            tokens.extend(self._apply_tokenizer(slot_values['text'], lang="multilingual"))

            # Tokenize after_text part
            tokens.extend(self._apply_tokenizer(after_text, lang=CANARY_SPECIAL_TOKENIZER))

            return tokens

        # For the user turn, we need to inject special tokenizer lang
        # so that control tokens are tokenized correctly.
        if "target_lang" in expected_slots and "source_lang" not in expected_slots:  # is user turn
            slot_values[self.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER

        return super().encode_turn(
            prompt_template=prompt_template, expected_slots=expected_slots, slot_values=slot_values
        )


@registered_prompt_format_fn(Cut, Canary2PromptFormatter)
def canary2(cut: Cut, prompt: Canary2PromptFormatter) -> dict[str, torch.Tensor]:
    """
    Prepend and append control tokens to the token sequence as per Canary 2.0 format.

    The prompt format syntax is defined in :class:`Canary2PromptFormatter`
    """
    if isinstance(cut, MixedCut):
        cut = cut._first_non_padding_cut
    if not isinstance(cut, MonoCut):
        raise TypeError(
            f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
        )
    if cut.custom is None:
        cut.custom = {}

    src_lang = cut.custom.get("source_lang")
    if src_lang is None:
        # Fallback to lhotse supervision language if not in manifest
        src_lang = cut.supervisions[0].language
        if src_lang is None:
            raise RuntimeError(f"Cut with id {cut.id} does not have language information.")

    is_translation = False
    # Use prompt.translation_task_prob if it exists
    if hasattr(prompt, "translation_task_prob") and prompt.translation_task_prob > 0.0:
        if "translation" in cut.supervisions[0].custom and cut.supervisions[0].custom["translation"] is not None:
            if random.random() < prompt.translation_task_prob:
                is_translation = True

    if is_translation:
        trgt_lang = "en"
        text = cut.supervisions[0].custom["translation"]
    else:
        trgt_lang = src_lang
        text = ' '.join(s.text for s in cut.supervisions if s.text is not None)

    # --- Construct user slots ---
    user_slots = {"decodercontext": cut.custom.get("decodercontext", "")}

    # Handle boolean slots
    for k in ("pnc", "itn", "diarize"):
        true_token = f"<|{k}|>"
        false_token = f"<|no{k}|>"

        # Default for pnc is yes, for others no.
        default_val = "yes" if k == "pnc" else "no"
        val = cut.custom.get(k, default_val)

        user_slots[k] = true_token if val in ("yes", "1", "True", "true", k) else false_token

    user_slots["emotion"] = cut.custom.get("emotion", "<|emo:undefined|>")
    user_slots["target_lang"] = f"<|{trgt_lang}|>"

    # --- Construct assistant slots ---
    assistant_slots = {
        "source_lang": f"<|{src_lang}|>",
        "text": text,
    }

    turns = [dict(role="user", slots=user_slots), dict(role="assistant", slots=assistant_slots)]
    ans = prompt.encode_dialog(turns)

    if isinstance(prompt.tokenizer, CanaryTokenizer) or isinstance(prompt.tokenizer, CanaryMultilingualTokenizer):
        eos = prompt.tokenizer.eos
    else:  # SPE
        eos = prompt.tokenizer.token_to_id(CANARY_EOS)  # type: ignore
    assert eos > -1, f"Invalid tokenizer: tokenizer.token_to_id('{CANARY_EOS}') returned {eos}"

    assert (
        ans["answer_ids"][-1].item() == eos
    ), f"Expected the last token in answer_ids to be EOS, but we got {ans['answer_ids']}"

    ans["answer_ids"] = ans["answer_ids"][:-1]  # Strip Canary's EOS
    return ans
