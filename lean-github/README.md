# InternLM-Step-Prover

<div align="center">

<img src="https://raw.githubusercontent.com/InternLM/InternLM/main/assets/logo.svg" width="200"/>
  <div> </div>
  <div align="center">
    <b><font size="5">InternLM-Math</font></b>
    <sup>
      <a href="https://internlm.intern-ai.org.cn/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    <div> </div>
  </div>

A state-of-the-art LEAN4 step prover.

[💻 Github](https://github.com/InternLM/InternLM-Math) [📊Dataset](https://huggingface.co/datasets/internlm/Lean-Github)
</div>

InternLM-Step-Prover is a 7B language model trained on Lean-Github and multiple sythesis datasets. InternLM-Step-Prover achieves state-of-the-art performances on MiniF2F, ProofNet, and Putnam math benchmarks, showing its formal math proving ability in multiple domains.

# Dialogue Example
```
### Input Example
DECL MyNat.mul_pow
GOAL a b n : N
⊢ (a * b) ^ n = a ^ n * b ^ n
### Output Example
PROOFSTEP induction n with t Ht
```

# Performance

## MiniF2F
| Method | Model size | Pass | miniF2F-valid | miniF2F-test |
|--------|------------|------|---------------|--------------|
| **Whole-Proof Generation Methods** |
| GPT-4-turbo 0409 | - | 64 | 25.4% | 23.0% |
| DeepSeekMath-Base | 7B | 128 | 25.4% | 27.5% |
| DeepSeek-Prover | 7B | 1 | - | 30.0% |
| | | 64 | - | 46.3% |
| | | 128 | - | 46.3% |
| | | 8192 | - | 48.8% |
| | | 65536 | - | 50.0% |
| | | cumulative | *60.2%* | *52.0%* |
| TheoremLlama | - | cumulative | 36.5% | 33.6% |
| **Tree Search Methods** |
| COPRA (GPT-3.5) | - | 1 | - | 9.0% |
| COPRA (GPT-4) | - | 1 | - | 26.6% |
| DSP(Isabelle) | 540B | 100 | 42.6% | 38.9% |
| Proof Artifact Co-Training | 837M | 1 | 23.9% | 24.6% |
| | | 8 | 29.3% | 29.2% |
| ReProver | 229M | 1 | - | 25.0% |
| Llemma | 7B | 1 | 26.2% | 26.2% |
| Llemma | 34B | 1 | 27.9% | 25.8% |
| Curriculum Learning | 837M | 1 | 33.6% | 29.6% |
| | | 8 | 41.2% | 34.5% |
| | | 64 | 47.3% | 36.6% |
| Hypertree Proof Search | 600M | cumulative | 58.6% | - |
| | | 64 | - | 41.0% |
| Lean-STaR | 7B | 64 | - | 46.3% |
| InternLM2-Math | 7B | 1 | 29.9% | 30.3% |
| InternLM2-Math-Plus | 7B | 1 | - | 43.4% |
| InternLM2-Step-Prover | 7B | 1 | 59.8% | 48.8% |
| InternLM2-Step-Prover | 7B | 64 | **63.9%** | **54.5%** |

## Proofnet & Putnam
| Method | Model size | Pass | result |
|--------|------------|------|--------|
| **ProofNet benchmark** |
| ReProver | 229M | 1 | 13.8% |
| InternLM2-Step-Prover | 7B | 1 | **18.1%** |
| **Putnam benchmark** |
| GPT-4 | - | 10 | 1/640 |
| COPRA (GPT-4) | - | 10 | 1/640 |
| DSP(Isabelle) | 540B | 10 | 4/640 |
| ReProver | 229M | 1 | 0/640 |
| InternLM2-Step-Prover | 7B | 1 | **5/640** |

# Citation and Tech Report
```
@misc{wu2024leangithubcompilinggithublean,
      title={LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover}, 
      author={Zijian Wu and Jiayu Wang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2407.17227},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.17227}, 
}
```
