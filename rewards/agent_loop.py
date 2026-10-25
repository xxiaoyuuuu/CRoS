"""verl agent loop shared by the rewards that probe the CoT (Early Answering, Adding Mistakes, CIR, SR).

It generates the rollout as usual (the only text that is trained on), then runs the reward's probes and hands
their results to the reward function (verl passes extra_fields["reward_scores"] as rollout_reward_scores).
"""

import asyncio
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from rewards.common import FROZEN_URL


class ProbeAgentLoop(AgentLoopBase):
    async def probe(self, prompt_ids, response, sampling_params, extra_info) -> dict:
        """Probe results of one rollout, for the reward function. Implemented by each reward."""
        raise NotImplementedError

    @property
    def max_len(self):  # prompt + response budget; a probe longer than this is skipped
        return self.rollout_config.prompt_length + self.rollout_config.response_length

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    async def generate(self, ids, sampling_params):
        """Generate with the policy being trained."""
        return await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=ids, sampling_params=sampling_params)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        n = self.rollout_config.response_length
        prompt_ids = await self.apply_chat_template(list(kwargs["raw_prompt"]))
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            out = await self.generate(prompt_ids, sampling_params)
        metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
        response_ids = out.token_ids[:n]
        try:
            results = await self.probe(prompt_ids, self.decode(response_ids), sampling_params,
                                       dict(kwargs.get("extra_info") or {}))
        except Exception:  # a failed probe must not stop training; the reward treats it as not measured
            results = {"failed": True}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=out.log_probs[:n] if out.log_probs else None,
            routed_experts=out.routed_experts[: len(prompt_ids) + n] if out.routed_experts is not None else None,
            num_turns=2,
            metrics=metrics,
            extra_fields=out.extra_fields,
        )
        output.extra_fields.update({"reward_scores": results, "turn_scores": [], "tool_rewards": []})
        return output


def rollout_sampling(sampling_params, max_tokens):
    """The rollout's own temperature / top_p / top_k, with another max_tokens."""
    defaults = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    sp = {k: d if sampling_params.get(k) is None else sampling_params[k] for k, d in defaults.items()}
    return dict(sp, max_tokens=max_tokens)


_frozen = {}


def frozen():
    """(async OpenAI client, semaphore) for the frozen base model; one per event loop."""
    loop = id(asyncio.get_running_loop())
    if loop not in _frozen:
        from openai import AsyncOpenAI
        _frozen[loop] = (AsyncOpenAI(base_url=FROZEN_URL, api_key="EMPTY", timeout=300, max_retries=2),
                         asyncio.Semaphore(64))
    return _frozen[loop]
