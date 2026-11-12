# Project guide

How the application works, what to show at the project defence, and how to extend it.

## 1. The problem each tool solves

**Datasheet Assistant.** Before writing a driver for a sensor, an embedded engineer reads its datasheet and copies the register addresses, bit fields, limits and scale factors by hand. It takes hours and errors are common (a wrong address, a reversed bit field). The tool does the copying with a language model and then *checks* the result with rules, because extraction errors are exactly what a model can make.

**Quality Report Analyzer.** At end-of-line test stations, operators write short free-text notes when a part fails ("noise test fail, rumble at high rpm", "torque high on fixture B"). Nobody has time to read hundreds of them, so patterns are seen late. The tool labels every note, groups them, computes where and when failures happen, reasons about likely causes and writes the shift-meeting report.

## 2. Why it is more than question-and-answer

| Processing step | Done by |
| --- | --- |
| page classification, extraction, labelling, naming, reasoning, writing | the model, each with its own prompt and parameters |
| PDF text with table rows kept (layout mode) | pypdf |
| schema validation of every answer, repair requests | pydantic + `structured.py` |
| duplicate / overlap / order checks | `checks.py` |
| C header | `cheader.py` (deterministic code) |
| retrieval of relevant pages | TF-IDF + cosine similarity (scikit-learn) |
| grouping of notes | TF-IDF + k-means, k chosen by silhouette score |
| Pareto, cross-tables, trend | pandas |

## 3. Parameter choices

| Step type | temperature | max_tokens | Reason |
| --- | --- | --- | --- |
| extraction, labelling | 0.0 | 4096 | the same input must give the same data; long JSON |
| cluster naming | 0.3 | 800 | a little variety in names is fine |
| root cause | 0.2 | 1500 | reasoning text plus JSON |
| brief, shift report | 0.7 | 2048 | readable, less repetitive prose; room for a reasoning model's thinking tokens |
| repair calls | 0.0 | same as the step | corrections must be exact |

The experiment (`experiments/param_tuning.py`) shows the effect: with higher temperature the valid-JSON rate and the agreement between runs fall, and too-small max_tokens cuts the JSON off, which the app reports as `TruncatedOutput` instead of crashing.

Measured with Groq (openai/gpt-oss-120b, 3 runs per temperature, `experiments/results/param_tuning.md`): temperature 0.0 to 0.4 gave valid, correct register maps in 3 of 3 runs; above 0.4 extraction became unreliable (0.8: 1 of 3 valid, 1.2: 2 of 3; the order of these two is chance with 3 runs). Labels stayed 100 % correct at every temperature. max_tokens 2048 completed; 512 and 128 ended in `TruncatedOutput`. This supports temperature 0 for extraction.

## 4. Error handling, end to end

| What goes wrong | What the user sees |
| --- | --- |
| no key / wrong key | "No API key for groq ..." / "The API key was rejected ..." |
| rate limit (429) | waits as long as the provider asks (retry-after header or "try again in 13.5s"), otherwise 10, 20, 40 s, with a "waiting N s" notice; a wait longer than 90 s (a daily limit) is reported at once; then "rate limit was hit even after waiting and retrying" |
| one request larger than the limit (413) | "larger than the provider allows ... Lower max_tokens in the sidebar" |
| timeout, provider offline | retried; then a clear message (for Ollama: "is `ollama serve` running?") |
| answer cut off | "The answer was cut off by max_tokens. Raise max_tokens in the sidebar." |
| invalid JSON or wrong fields | repaired by the model (up to 2 times), then "returned invalid data after 2 repair attempts" |
| scanned / encrypted / non-PDF file | "The PDF has (almost) no text layer ..." etc. |
| CSV with missing columns, duplicate ids, bad dates | which column or value is wrong |
| a step fails but others can continue | the run continues and shows a warning (for example no scale factors) |
| demo mode with another file | "Demo mode replays recorded answers for the sample files only ..." |

## 5. Defence: a 5-minute demo

1. Start in **demo** mode, sample datasheet, *Analyse datasheet*: show 2 problems found and fixed, the register table, the header download, the brief.
2. Ask "What is the I2C address?" and point at the page citations.
3. Quality tab, sample notes: Pareto, the ST3 night-shift pattern, the root-cause expander, the report.
4. *Prompts and settings used* tab: the technique and parameters of every call, and the exact prompt text.
5. Switch to **groq** with your key and run the datasheet again: real model, same pipeline. Change the extraction temperature to 1.2 and show the difference.

Likely questions and short answers:

| Question | Answer |
| --- | --- |
| Why few-shot for extraction? | The example fixes the JSON format and the hex/bit notation, which zero-shot gets wrong more often. |
| Why chain-of-thought for scale factors? | It is a multi-step calculation (find field, list codes, 32768 / range, check halving); reasoning first improves the final numbers. |
| Why is the header written by code, not by the model? | So every `#define` comes from validated data and is reproducible; the model only extracts. |
| How do you know the output is correct? | pydantic schemas, rule checks, and the samples' known correct answers (accuracy is shown in the app and measured by the experiment). |
| Why temperature 0 for extraction? | Data must be repeatable; creativity only helps prose. |
| What does the persona change? | Expertise and audience: the reviewer is strict and changes only what the datasheet supports; the shift report is short and action-oriented. |

## 6. Extending it

- Another provider: add it to `PROVIDERS` and `_chat_model` in `assistant/llm.py`.
- Another output (for example a Python driver skeleton): add a function next to `cheader.py` that reads the checked `RegisterMap`.
- Real data: export failure notes with the columns `note_id, date, shift, station, product, serial_no, note`.
- New demo answers after changing a prompt: edit `tools/make_samples.py` and run it.
