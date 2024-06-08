# `InternLM-Math` MiniF2F performance reproduce 

Scripts for the Lean formal2formal (tactic prediction) experiments. Adapted from
[llemma-formal2formal](https://github.com/wellecks/llemma_formal2formal).


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

#### Run
```
cd scripts
bash eval_internLM2-plus_7b.sh
```

#### About Lean4 version
The scripts fetch MiniF2F from [https://github.com/rah4927/lean-dojo-mew], which uses leanprover/lean4:nightly-2023-08-19. To use newer versions of Lean4, slight modifications may be needed. We provide a version of MiniF2F that works with leanprover/lean4:v4.7.0 at [https://github.com/wzj423/lean-dojo-mew]. To use a custom source for the MiniF2F dataset, change the `url` and `commit` fields in the minif2f/data/minif2f.jsonl file.
