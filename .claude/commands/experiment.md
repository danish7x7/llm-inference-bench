# Run and document a benchmark experiment

You are helping run a benchmark experiment on the LLM inference suite and turn
the result into a documented writeup. Follow this sequence — do NOT skip the
"stop and wait" steps, because you cannot see GPU output.

## Arguments
The user will describe the experiment in their message (e.g. "batch sweep on
tinyllama", "fp16 vs awq on a 7B"). If the experiment is ambiguous about
model, runner, prompt set, or what's being varied, ask ONE clarifying question
before proceeding.

## Steps
1. State the hypothesis. One or two sentences: what this measures and what you
   expect to see, tied to inference mechanics.
2. Give the exact `python benchmark.py ...` command. Note results land in
   results/<runner>_<model>_<timestamp>.csv. Then STOP and ask the user to run
   it and paste the output. Do not continue until they do.
3. Read the result: extract per-config metrics (TTFT, ITL, throughput, VRAM)
   into a clean table; compute relative changes vs baseline.
4. Interpret honestly in terms of serving mechanics (batching, KV cache,
   prefill vs decode, quantization tradeoff). Name the tradeoff explicitly.
5. State caveats: small/repeated prompt sets, prefix caching, tiny model, VRAM
   ceiling, single sample. Name them before a reviewer would.
6. Write docs/<experiment-name>.md: hypothesis, setup (model, hardware, flags,
   versions), table, interpretation, caveats, one-line "what's next". Keep it
   tight — portfolio material, not a lab dump.

## Reminders
- 8GB RTX 4060 under WSL. Stay within those limits.
- Don't fabricate numbers — every number must come from pasted output.
- The interpretation is the valuable part; that's what shows understanding.
