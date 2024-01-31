# `InternMath2` formal2formal 

Scripts for the Lean formal2formal (tactic prediction) experiments. Adapted from
[`llemma`_formal2formal](https://github.com/wellecks/llemma_formal2formal).


#### Setup
Install Python packages:
```
pip install -r requirements.txt
```

Install Lean:
```
# from https://leanprover-community.github.io/install/linux.html
# After running this command, select (2), then `nightly`, then `y`:
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env
lake
```

Configure LeanDojo:
```
export CONTAINER="native"
```

Patch vllm

Currently vllm doesn't officially support `internLM2`. Adaptations from the vllm community can be found here [#2666](https://github.com/vllm-project/vllm/pull/2666) [#2527](https://github.com/vllm-project/vllm/pull/2527).


#### Run
See `scripts`

#### Compute metrics

```bash
python compute_metrics.py
==>

internLM2-7b_minif2f_test       0.30327868852459017     74      244
internLM2-20b_minif2f_test      0.29508196721311475     72      244
```




