import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS = 128


class VLLMModel:
    def __init__(
        self,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        dtype="half",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=True,
    ):
        from vllm import LLM, SamplingParams

        self.model_name = model
        self.max_tokens = max_tokens
        self.llm = LLM(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    def warmup(self, prompts):
        self.run(prompts)

    def run(self, prompts):
        t0 = time.time()
        outs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        dt = time.time() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outs)
        return total_out / dt, dt, total_out

    def time_batch(self, prompts):
        t0 = time.time()
        self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return time.time() - t0
