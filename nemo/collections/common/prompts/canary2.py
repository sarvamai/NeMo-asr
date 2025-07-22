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
            "template": f"{CANARY2_BOCTX}|decodercontext|{CANARY_BOS}",
            "slots": {
                # Empty string or previous transcript / other context to bias predictions.
                "decodercontext": Modality.Text,
            },
        },
        # System's reponse.
        OUTPUT_ROLE: {
            "template": f"|emotion||source_lang||target_lang||pnc||itn||timestamp||diarize||text|{CANARY_EOS}",
            "slots": {
                # Emotion of the speaker - may be predicted by the model with a partial prompt.
                "emotion": Modality.TextLiteral(
                    "<|emo:undefined|>", "<|emo:neutral|>", "<|emo:angry|>", "<|emo:happy|>", "<|emo:sad|>"
                ),
                # Audio input language - may be predicted by the model with a partial prompt.
                "source_lang": Modality.Text,
                # Transcription language - specified by the user.
                "target_lang": Modality.Text,
                # Should we predict punctuation and capitalization?
                "pnc": Modality.TextLiteral(*(PNC_TRUE | PNC_FALSE)),
                # Should we predict with inverse text normalization (numerals as digits, abbreviations, etc.)
                "itn": Modality.TextLiteral(*(ITN_TRUE | ITN_FALSE)),
                # Should we predict timestamps?
                "timestamp": Modality.TextLiteral(*(TIMESTAMP_TRUE | TIMESTAMP_FALSE)),
                # # Should we diarize speech?
                "diarize": Modality.TextLiteral(*(DIARIZE_TRUE | DIARIZE_FALSE)),
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
                    prompt = prompt.replace(_mangled(slot), slot_values[slot])
            tokens.extend(self._apply_tokenizer(prompt, lang=CANARY_SPECIAL_TOKENIZER))

            # Tokenize text part
            text_lang = slot_values.get('text_lang') or slot_values.get('target_lang')
            tokens.extend(self._apply_tokenizer(slot_values['text'], lang=text_lang))

            # Tokenize after_text part
            tokens.extend(self._apply_tokenizer(after_text, lang=CANARY_SPECIAL_TOKENIZER))

            return tokens

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


    expected_slots = {"source_lang", "target_lang"}
    missing_keys = expected_slots - set(cut.custom)
    if missing_keys:
        # fix for missing source_lang
        language = cut.supervisions[0].language
        language = language.lower()
        # add to cut.custom
        # TODO: This needs to be fixed as this won't work for the translation task
        cut.custom["source_lang"] = cut.supervisions[0].custom['source_lang']
        cut.custom["target_lang"] = cut.supervisions[0].custom['target_lang']

    # first, validate the utterance
    expected_slots = {"source_lang", "target_lang"}
    missing_keys = expected_slots - set(cut.custom)
    if missing_keys:
        raise RuntimeError(
            f"We found cut with ID {cut.id} that is missing the following keys: {missing_keys}"
            f"Please ensure that every utterance in the input manifests contains these keys."
        )
        
        
    src_lang = cut.custom.get("source_lang")
    trgt_lang = cut.custom.get("target_lang")
    
    # an-thony start
    is_translation = False
    if prompt.translation_task_prob > 0.0 and "translation" in cut.supervisions[0].custom:
        if random.random() < prompt.translation_task_prob:
            is_translation = True
            trgt_lang = "en"
            text = cut.supervisions[0].custom["translation"]
        else:
            text = ' '.join(s.text for s in cut.supervisions if s.text is not None)
    else:
        text = ' '.join(s.text for s in cut.supervisions if s.text is not None)
    
    cut.custom['target_lang'] = trgt_lang
    # an-thony end
    
    user_slots = {"decodercontext": cut.custom.get("decodercontext", "")}

    optional_slots = {
        "emotion": "<|emo:undefined|>",
        "itn": "<|noitn|>",
        "timestamp": "<|notimestamp|>",
        "diarize": "<|nodiarize|>",
        "pnc": "<|pnc|>",  # consistent with canary1
    }

    assistant_slots = {
        "source_lang": f"<|{src_lang}|>",
        "target_lang": f"<|{trgt_lang}|>",
        "text": text,
        "text_lang": ifnone(cut.supervisions[0].language, trgt_lang),
    }

    for k, v in optional_slots.items():
        assistant_slots[k] = cut.custom.get(k, v)

    turns = [dict(role="user", slots=user_slots), dict(role="assistant", slots=assistant_slots)]
    # If data has no transcript, create empty response with <eos> only.
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
