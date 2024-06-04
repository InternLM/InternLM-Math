# InternLM-Math Agent

The performance of InternLM-Math could be lifted by external tools. We provide an example using [Lagent](https://github.com/InternLM/lagent) to build agents interacting with the code interpreter to solve MATH problems.

## Setup

Install the extra dependencies

```sh
pip install -r requirements.txt
```

## Usages

Run the following script to perform inference and evaluation on MATH test.


```sh
python math_agent.py \
  --backend=lmdeploy \  # For HuggingFace models: hf
  --model_path=internlm/internlm2-math-plus-1_8b \
  --tp=1 \
  --temperature=1.0 \
  --top_k=1 \
  --output_path=math_lmdeploy.jsonl \
  --do_eval
```

`output_path` is a jsonl format file to save the inference results. Each line is like

```json
{
  "problem": "The point $(a, b)$ lies on the line with the equation $3x + 2y = 12.$ When $a = 4$, what is the value of $b$?",
  "level": "Level 2",
  "type": "Algebra",
  "solution": "We plug in $x = 4$: \\begin{align*}\n3(4) + 2y &= 12\\\\\n12 + 2y &= 12\\\\\ny &= 0.\n\\end{align*}\n\nTherefore, $b = \\boxed{0}$.",
  "idx": 41,
  "query": "The point $(a, b)$ lies on the line with the equation $3x + 2y = 12.$ When $a = 4$, what is the value of $b$?",
  "gt": "0",
  "pred": [
    "0"
  ],
  "steps": [
    {
      "role": "language",
      "content": "To find the value of $b$ when $a = 4$ for the line with the equation $3x + 2y = 12$, we can substitute $a = 4$ into the equation and solve for $b$. This will give us the coordinates of the point $(a, b)$ on the line.\n\nLet's start by substituting $a = 4$ into the equation and solving for $b$."
    },
    {
      "role": "tool",
      "content": {
        "name": "IPythonInteractiveManager",
        "parameters": {
          "command": "```python\nfrom sympy import symbols, Eq, solve\n\n# Define symbols\na, b = symbols('a b')\n\n# Given equation of the line\nequation = Eq(3*a + 2*b, 12)\n\n# Substitute a = 4 into the equation\nsubstituted_equation = equation.subs(a, 4)\n\n# Solve for b\nsolution = solve(substituted_equation, b)\n\nsolution[0]\n```"
        }
      },
      "name": "interpreter"
    },
    {
      "role": "environment",
      "content": "0",
      "name": "interpreter"
    },
    {
      "role": "language",
      "content": "After substituting $a = 4$ into the equation and solving for $b$, we find that $b = 0$. Therefore, the value of $b$ when $a = 4$ for the given line is $\\boxed{0}$."
    }
  ],
  "error": null
}
```

Once it is prepared, just skip the inference stage as follows.

```bash
python math_agent.py \
  --output_path=math_lmdeploy.jsonl \
  --no-do_infer \
  --do_eval
```

Please refer to [`math_agent.py`](math_agent.py) for more information about the arguments.