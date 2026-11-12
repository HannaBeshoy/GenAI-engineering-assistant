# Parameter tuning (groq, openai/gpt-oss-120b, 3 runs per temperature)

| temperature | valid JSON (registers) | register F1 | field F1 | valid JSON (labels) | label accuracy |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 3/3 | 1.0 | 1.0 | 3/3 | 1.0 |
| 0.4 | 3/3 | 1.0 | 1.0 | 3/3 | 1.0 |
| 0.8 | 1/3 | 0.333 | 0.333 | 3/3 | 1.0 |
| 1.2 | 2/3 | 0.667 | 0.667 | 3/3 | 1.0 |

| max_tokens | result |
| --- | --- |
| 2048 | complete answer |
| 512 | TruncatedOutput: The model stopped at max_tokens=512 before finishing; 60 of those tokens went to its hidden reasoning before the answer. |
| 128 | TruncatedOutput: The model stopped at max_tokens=128 before finishing; 60 of those tokens went to its hidden reasoning before the answer. |
