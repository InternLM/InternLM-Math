MAX_ITERS=4000
NUM_SAMPLES=32

TEMPERATURES="0.0" # For pass@1 test, beam-search(t=0.0) is often the most performant choice. If pass@32 is wanted, change temperature to 0.7 or 0.9.

TIMEOUT=1500
NUM_SHARDS=8
DATASET="minif2f-test"
DATA="data/minif2f-lean4.7.0.jsonl"

MODEL="internlm/internlm2-step-prover"
NAME="internLM2-step-prover"

OUTPUT_DIR="output/${NAME}_minif2f_test"
mkdir -p logs
for SHARD in 0 1 2 3 4 5 6 7
do
  CUDA_VISIBLE_DEVICES=${SHARD} python proofsearch_internLM2-plus.py --dataset-name ${DATASET} \
  --temperatures ${TEMPERATURES} --timeout ${TIMEOUT} --num-shards ${NUM_SHARDS} \
  --shard ${SHARD} --model-name ${MODEL} --max-iters ${MAX_ITERS} --dataset-path ${DATA} \
  --num-samples ${NUM_SAMPLES} --early-stop --output-dir ${OUTPUT_DIR} \
  &> logs/${NAME}_shard${SHARD}.out &
done