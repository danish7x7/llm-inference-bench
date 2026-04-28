from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )

    prompts = ["Hello, my name is", "The capital of France is"]
    params = SamplingParams(max_tokens=20, temperature=0.0)

    outputs = llm.generate(prompts, params)
    for o in outputs:
        print(f"PROMPT: {o.prompt!r}")
        print(f"OUTPUT: {o.outputs[0].text!r}")
        print("---")

if __name__ == "__main__":
    main()
