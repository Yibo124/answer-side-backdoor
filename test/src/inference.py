# Third-party notice: template formats and segmented encoding are adapted from
# LLaMA Factory (https://github.com/hiyouga/LLaMA-Factory), specifically
# src/llamafactory/data/template.py and src/llamafactory/data/formatter.py.
# Copyright 2025 the LlamaFactory team.
#
# Modified for standalone Transformers/PEFT text inference, with training,
# tool-calling, and multimodal support removed.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use the adapted code except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone Transformers/PEFT inference with explicit text chat templates.

Heavy dependencies are imported only when loading a tokenizer/model or generating.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional


SUPPORTED_TEMPLATES = ("llama2", "mistral", "gemma", "deepseekr1", "default")


def hub_cache_dir() -> Path:
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "huggingface" / "hub"


def resolve_model_name_or_path(value: str, revision: Optional[str] = None) -> str:
    """Resolve local paths and cached snapshots, or return an uncached Hub ID."""
    hub = hub_cache_dir()
    path = Path(value).expanduser()
    if path.exists():
        root = path
    else:
        match = re.search(r"(?:^|/)(models--[^/]+)(?:/snapshots/([^/]+))?", value)
        if match:
            root = hub / match.group(1)
            revision = revision or match.group(2)
            repo_id = match.group(1)[len("models--"):].replace("--", "/", 1)
        elif not path.is_absolute() and "/" in value:
            root = hub / ("models--" + value.replace("/", "--"))
            repo_id = value
        else:
            raise FileNotFoundError(f"Model directory not found: {value}. Pass a local model path or a Hub ID.")
        if not root.exists():
            if path.is_absolute():
                raise FileNotFoundError(f"Moved model {value} is absent from {hub}. Pass --model-name-or-path {repo_id}.")
            # Transformers handles authentication and downloads for uncached IDs.
            return repo_id

    if (root / "config.json").is_file():
        return str(root.resolve())
    ref = root / "refs" / (revision or "main")
    commit = ref.read_text().strip() if ref.is_file() else revision
    if commit:
        snapshot = root / "snapshots" / commit
        if (snapshot / "config.json").is_file():
            return str(snapshot.resolve())
        raise FileNotFoundError(f"Model revision {commit!r} is not cached in {root}.")
    snapshots = sorted(p for p in (root / "snapshots").glob("*") if (p / "config.json").is_file())
    if len(snapshots) == 1:
        return str(snapshots[0].resolve())
    raise ValueError(f"Cannot select a model snapshot in {root}; specify --model-revision or a snapshot path.")


def resolve_adapter_name_or_path(value: str) -> str:
    """Resolve comma-separated local adapter paths or names under test/saves/."""
    saves_dir = Path(__file__).resolve().parents[1] / "saves"
    resolved = []
    for item in value.split(","):
        path = Path(item.strip()).expanduser()
        candidate = path if path.exists() else saves_dir / path
        if not (candidate / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter not found: {item!r}; expected a directory under {saves_dir} or a local path.")
        resolved.append(str(candidate.resolve()))
    return ",".join(resolved)


def infer_template_from_model_path(value: str) -> str:
    name = value.lower()
    if "deepseek-r1-distill-llama" in name:
        return "deepseekr1"
    if "mistral-7b-instruct" in name:
        return "mistral"
    if "llama-2" in name or "llama2" in name:
        return "llama2"
    if "gemma-7b" in name:
        return "gemma"
    config = Path(value) / "config.json"
    if config.is_file():
        kind = json.loads(config.read_text()).get("model_type")
        if kind == "gemma":
            return "gemma"
        if kind == "mistral":
            return "mistral"
    raise ValueError(f"Cannot infer the chat template for {value!r}; pass --template explicitly.")


class TextTemplate:
    """Encode template segments with explicitly placed BOS/EOS tokens."""

    def __init__(self, name: str):
        if name not in SUPPORTED_TEMPLATES:
            raise ValueError(f"Unsupported template {name!r}; supported: {', '.join(SUPPORTED_TEMPLATES)}")
        self.name = name

    def fix_tokenizer(self, tokenizer: Any) -> None:
        if self.name == "gemma":
            token = "<end_of_turn>"
            if token not in tokenizer.get_vocab():
                raise ValueError("Gemma tokenizer is missing <end_of_turn>.")
            tokenizer.eos_token = token
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define an EOS token; refusing to add untrained tokens.")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def get_stop_token_ids(self, tokenizer: Any) -> list[int]:
        # Accept Gemma's native EOS and chat turn terminator so generation stops with either the base model or the evaluated LoRA adapters.
        if self.name == "gemma":
            native_eos = tokenizer.convert_tokens_to_ids("<eos>")
            return list(dict.fromkeys([tokenizer.eos_token_id, native_eos]))
        return [tokenizer.eos_token_id]

    def encode_prompt(self, tokenizer: Any, messages: list[dict[str, str]], system: str = "") -> list[int]:
        """Encode a conversation ending with the user, ready for an assistant reply."""
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Generation history must end with a user message.")
        ids: list[int] = []
        # Encode each template segment separately to preserve token boundaries.
        def text(value: str) -> None:
            if value:
                ids.extend(tokenizer.encode(value, add_special_tokens=False))
        def bos() -> None:
            if tokenizer.bos_token_id is not None:
                ids.append(tokenizer.bos_token_id)
        for index, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("Messages must alternate user/assistant.")
            if index == 0 and self.name in {"mistral", "gemma", "deepseekr1"}:
                bos()
            if index == 0 and system:
                if self.name == "llama2":
                    content = f"<<SYS>>\n{system}\n<</SYS>>\n\n" + content
                elif self.name in {"mistral", "gemma"}:
                    content = system + "\n\n" + content
                elif self.name == "deepseekr1":
                    text(system)
                else:
                    text(f"System: {system}")
                    ids.append(tokenizer.eos_token_id)
                    text("\n")
            if role == "user":
                if self.name == "llama2":
                    bos()
                    text(f"[INST] {content} [/INST]")
                elif self.name == "mistral":
                    text(f"[INST] {content}[/INST]")
                elif self.name == "gemma":
                    text(f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n")
                elif self.name == "deepseekr1":
                    text(f"<｜User｜>{content}<｜Assistant｜>")
                else:
                    text(f"Human: {content}")
                    ids.append(tokenizer.eos_token_id)
                    text("\nAssistant:")
            else:
                if self.name == "gemma":
                    text(content + "<end_of_turn>\n")
                else:
                    if self.name == "deepseekr1":
                        content = re.sub(r"<think>\n.*?\n</think>\n\n", "", content, flags=re.DOTALL).lstrip("\n")
                    text((" " if self.name == "mistral" else "") + content)
                    ids.append(tokenizer.eos_token_id)
                    if self.name == "default":
                        text("\n")
        if self.name == "deepseekr1":
            # Request a direct answer with an empty, closed reasoning block, matching LLaMA Factory's enable_thinking=False prefix.
            text("<think>\n\n</think>\n\n")
        return ids


def load_tokenizer(source: str, template: TextTemplate, *, cache_dir: Optional[str] = None,
                   revision: Optional[str] = None, trust_remote_code: bool = True) -> Any:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.utils.hub import cached_file
    from tokenizers import Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, padding_side="left",
        cache_dir=cache_dir, revision=revision, trust_remote_code=trust_remote_code)
    # Resolve the original tokenizer file for local paths and Hub IDs, including first-time downloads.
    tokenizer_file = cached_file(source, "tokenizer.json", cache_dir=cache_dir, revision=revision,
        _raise_exceptions_for_missing_entries=False)
    if tokenizer_file is not None:
        serialized = Tokenizer.from_file(str(tokenizer_file))
        # Restore the serialized ByteLevel decoder if auto-detection selected another decoder. Preserve already-correct tokenizers, including Gemma's decoder for spaces and byte fallback.
        if "ByteLevel" in str(serialized.decoder) and "ByteLevel" not in str(tokenizer.backend_tokenizer.decoder):
            old = tokenizer
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file),
                bos_token=old.bos_token, eos_token=old.eos_token, pad_token=old.pad_token,
                unk_token=old.unk_token, padding_side="left")
    template.fix_tokenizer(tokenizer)
    tokenizer.padding_side = "left"
    return tokenizer


class BaseBatchInferencer:
    def generate_batch(self, batch_messages, systems, progress_desc=None) -> list[str]:
        raise NotImplementedError


class HuggingFaceBatchInferencer(BaseBatchInferencer):
    def __init__(self, infer_args: dict[str, Any], batch_size: int):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
        self.torch = torch
        self.batch_size = batch_size
        self.template = TextTemplate(infer_args["template"])
        model_path = infer_args["model_name_or_path"]
        common = dict(cache_dir=str(hub_cache_dir()), revision=infer_args.get("model_revision"),
                      trust_remote_code=infer_args["trust_remote_code"])
        # Use the base model's tokenizer for both base and LoRA evaluation.
        self.tokenizer = load_tokenizer(model_path, self.template, **common)
        config = AutoConfig.from_pretrained(model_path, **common)
        dtype_name = infer_args["infer_dtype"]
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config, dtype=dtype,
            device_map={"": device}, attn_implementation="sdpa", **common)
        adapters = infer_args.get("adapter_name_or_path")
        if adapters:
            from peft import PeftModel
            for adapter in adapters.split(","):
                self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
                self.model = self.model.merge_and_unload()
        self.model.requires_grad_(False)
        self.model.eval()
        if max(self.tokenizer.get_vocab().values()) >= self.model.get_input_embeddings().num_embeddings:
            raise ValueError("Tokenizer token IDs exceed model vocabulary size.")
        self.cutoff_len = infer_args["cutoff_len"]
        self.max_new_tokens = infer_args["max_new_tokens"]
        self.context_window = getattr(config, "max_position_embeddings", None)
        generation_kwargs = dict(do_sample=True, temperature=infer_args["temperature"],
            top_p=infer_args["top_p"], top_k=infer_args["top_k"], num_beams=1,
            max_new_tokens=self.max_new_tokens, repetition_penalty=infer_args["repetition_penalty"],
            length_penalty=1.0, eos_token_id=self.template.get_stop_token_ids(self.tokenizer),
            pad_token_id=self.tokenizer.pad_token_id)
        if not generation_kwargs["temperature"]:
            generation_kwargs["do_sample"] = False
            generation_kwargs.pop("temperature")
            generation_kwargs.pop("top_p")
        self.generation_config = GenerationConfig(**generation_kwargs)
        print(f"Loaded {type(self.model).__name__}: dtype={self.model.dtype}, tokenizer={type(self.tokenizer).__name__}, "
              f"template={self.template.name}, EOS={self.generation_config.eos_token_id}, context={self.context_window}")

    def generate_batch(self, batch_messages, systems, progress_desc=None) -> list[str]:
        from tqdm import tqdm
        outputs: list[str] = []
        for start in tqdm(range(0, len(batch_messages), self.batch_size), desc=progress_desc or "Inference", leave=False):
            chunk = batch_messages[start:start+self.batch_size]
            chunk_systems = systems[start:start+self.batch_size]
            full_ids = [self.template.encode_prompt(self.tokenizer, m, s) for m, s in zip(chunk, chunk_systems)]
            # Keep the last cutoff_len input tokens, independently of the output budget.
            prompt_ids = [ids[-self.cutoff_len:] for ids in full_ids]
            width = max(map(len, prompt_ids))
            input_ids = self.torch.full((len(prompt_ids), width), self.tokenizer.pad_token_id,
                dtype=self.torch.long, device=self.model.device)
            attention_mask = self.torch.zeros_like(input_ids)
            for row, ids in enumerate(prompt_ids):
                input_ids[row, -len(ids):] = self.torch.tensor(ids, device=self.model.device)
                attention_mask[row, -len(ids):] = 1
            with self.torch.inference_mode():
                generated = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                    generation_config=self.generation_config)
            response_ids = generated[:, width:].tolist()
            for response in response_ids:
                # Batched generation may pad finished rows; discard tokens after EOS.
                stop = next((i for i, token in enumerate(response) if token in self.generation_config.eos_token_id), None)
                if stop is not None:
                    response = response[:stop+1]
                # Retain cleanup=True to preserve the evaluation's decoding behavior.
                output = self.tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                outputs.append(output)
        return outputs
