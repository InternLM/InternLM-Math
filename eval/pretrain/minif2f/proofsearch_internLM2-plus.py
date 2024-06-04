# Lean proof search with LeanDojo interaction
# Author: Sean Welleck
import json
import heapq
import subprocess
import time
import transformers
import vllm
from datetime import datetime
from lean_dojo import *
from pathlib import Path
from tqdm import tqdm, trange

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def chat_template_to_prompt(prompt_list):
    result = ""
    total_step = len(prompt_list)
    for i, message in enumerate(prompt_list):
        result += ('[UNUSED_TOKEN_146]' + message['role'] +
                   '\n' + message['content'])
        if i+1 != total_step:
            result += '[UNUSED_TOKEN_145]\n'
        elif message['role'] == 'user':
            result += '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
    return result

def prompt_style_internlm_chat_0522_extractor(result:str):
    START_STR="Here is the predicted next tactic:\n```lean\n"
    END_STR="\n```"
    if result.startswith(START_STR):
        result=result[len(START_STR):]
    if result.endswith(END_STR):
        result=result[:-len(END_STR)]
    return result

def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):
    if not isinstance(prompt, str):
        prompt = chat_template_to_prompt(prompt)
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
            max_tokens=max_tokens,
            stop=stop,
        )
        outputs = model.generate([prompt], params, use_tqdm=False)
        if len(outputs) == 0:
            return [], []
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts = list(map(prompt_style_internlm_chat_0522_extractor,texts))
    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts


def _prompt_fewshot(state):
    prompt = f"My LEAN 4 state is:\n```lean\n" + state + \
        "```\nPlease predict a possible tactic to help me prove the theorem."
    prompt = [{"role": "user", "content": prompt}]
    return prompt


def best_first_search(
        theorem,
        model,
        tokenizer,
        max_iters,
        temperatures,
        num_samples,
        prompt_fn,
        timeout=600,
        early_stop=False,
        max_tokens=256
) -> dict:
    """Best first search."""
    attempt_results = []
    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):

            start = time.time()
            proof_finished = False
            queue = [(0.0, [], init_state, [])]
            visited = set()

            for iteration in trange(max_iters):
                if len(queue) == 0 or proof_finished:
                    break

                total_score, steps, state, trace = heapq.heappop(queue)
                ts = _tactic_state(state)
                visited.add(ts)

                step_cands, step_scores = generate_vllm(
                    prompt_fn(ts),
                    model,
                    tokenizer,
                    temperatures,
                    num_samples,
                    stop=['[UNUSED_TOKEN_145]',],
                    max_tokens=max_tokens
                )
                step_cands = [s.strip() for s in step_cands]

                for step, score in zip(step_cands, step_scores):
                    result = dojo.run_tac(state, step)
                    step_trace = {
                        "tactic": step,
                        "state_before": _tactic_state(state)
                    }
                    if isinstance(result, ProofFinished):
                        attempt_results.append({
                            'theorem': theorem.full_name,
                            'proof': steps + [step],
                            'score': total_score - score,
                            'success': True,
                            'failure_reason': '',
                            'trace': trace + [step_trace],
                            'temperature': temperatures,
                            'elapsed': start - time.time(),
                            'iteration': iteration
                        })
                        if early_stop:
                            return attempt_results
                        proof_finished = True
                        break
                    elif isinstance(result, TacticState):
                        if _tactic_state(result) not in visited:
                            # Score is negative log probability summed across steps
                            new_score = (total_score - score)
                            heapq.heappush(
                                queue, (new_score, steps+[step], result, trace+[step_trace])
                            )
    except (DojoInitError, DojoHardTimeoutError, DojoCrashError, subprocess.CalledProcessError) as e:
        if len(attempt_results) == 0:
            attempt_results.append({
                'theorem': theorem.full_name,
                'success': False,
                'failure_reason': type(e).__name__
            })

    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem.full_name,
            'success': False,
            'failure_reason': 'SearchEnded'
        })

    return attempt_results


def _save(model_name, results, args_dict, output_dir, shard):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        'results__%s__%s.json' % (model_name.replace('/', '_'), shard)
    )
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'args': args_dict
            }, f, indent=4)
        print(output_file)


def _load_model(model_name, tp_degree):
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_degree,
        dtype='bfloat16',
        max_num_batched_tokens=8192,
        trust_remote_code=True,
        enforce_eager=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    return model, tokenizer


def _load_data(dataset_name, dataset_path):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    else:
        raise NotImplementedError(dataset_name)

    return repo, data


def print_stats(results):
    print(len([x for x in results if x['success']]) / len(results))
    print("# successes: ", len([x for x in results if x['success']]), sep="\t")


def resume_from(results_filename, data):
    results = json.load(open(results_filename))['results']
    data = data[len(results):]
    print("=== Resuming from %d" % (len(results)))
    return results, data


def make_output_dir(output_dir):
    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name', 
        choices=[
            'internlm/internlm2-math-base-7b',
            "internlm/internlm2-math-base-20b",
            "internlm/internlm2-math-plus-7b",
            "internlm/internlm2-math-plus-20b"
        ],
        required=True
    )
    parser.add_argument(
        '--dataset-name',
        default='minif2f-test',
        choices=['minif2f-valid', 'minif2f-test']
    )
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--dataset-path', default='data/minif2f.jsonl')
    parser.add_argument('--output-dir', default='output/minif2f')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--tp-degree', type=int, default=1)
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--num-examples', type=int, default=-1)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--clear-process-hours', type=int, default=3)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.0])
    args = parser.parse_args()
    model, tokenizer = _load_model(args.model_name, args.tp_degree)
    output_dir = make_output_dir(args.output_dir)

    repo, data = _load_data(args.dataset_name, args.dataset_path)
    shard_size = len(data) // args.num_shards
    data = data[args.shard*shard_size:(args.shard+1)*shard_size] if args.num_shards > 1+ args.shard else data[args.shard*shard_size:]
    print("Shard size: %d" % (len(data)))

    if args.resume_from is not None:
        results, data = resume_from(args.resume_from, data)
    else:
        results = []

    start = time.time()
    for example in tqdm(data, total=len(data)):
        file_path = example['file_path']
        theorem_name = example['full_name']
        theorem = Theorem(repo, file_path, theorem_name)
        attempt_results = best_first_search(
            theorem, model, tokenizer,
            max_iters=args.max_iters,
            prompt_fn=_prompt_fewshot,
            temperatures=args.temperatures,
            num_samples=args.num_samples,
            timeout=args.timeout,
            early_stop=args.early_stop
        )

        result = {
            'attempt_results': attempt_results,
            'success': any([x['success'] for x in attempt_results]),
            'example': example
        }

        results.append(result)

        _save(
            model_name=args.model_name,
            results=results,
            args_dict=args.__dict__,
            output_dir=output_dir,
            shard=args.shard
        )
        print_stats(results)
        # The proof search occasionally leaves Lean processes open. As a workaround,
        # we periodically kill all Lean processes. Note that this may cause a proof search failure.
        if args.shard == 0:
            hours = 60*60*args.clear_process_hours
            if time.time() - start > hours:
                print("=== Killing active leanprover processes to mitigate leak")
                os.system("ps aux | grep leanprover | awk '{print $2}' | xargs kill -9")
                start = time.time()
