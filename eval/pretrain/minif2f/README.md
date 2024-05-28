# `InternLM-Math` MiniF2F performance reproduce 

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

#### Run
```
cd scripts
bash eval_internLM2-plus_7b.sh
```
