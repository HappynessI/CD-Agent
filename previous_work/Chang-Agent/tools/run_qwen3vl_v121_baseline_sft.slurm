#!/usr/bin/env bash
#SBATCH --job-name=CA0801V121SFT
#SBATCH --partition=gre
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT required}")";WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT required}")"
RUN_DIR="$(realpath -e -- "${1:?usage: worker RUN_DIR}")";DATASET_DIR="$(realpath -e -- "${DATASET_DIR:?DATASET_DIR required}")"
TRAIN_ENV="${TRAIN_ENV:-$WORKSPACE/conda_envs/qwen3vl-sft-env}";PYTHON="$TRAIN_ENV/bin/python";BASE_MODEL="$WORKSPACE/models/Qwen3-VL-2B-Instruct"
LLAMA_FACTORY_ROOT="$WORKSPACE/Chang-Agent/third_party/LLaMA-Factory";LLAMA_COMMIT="9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d";CONFIG="$ROOT/configs/qwen3vl_v121_baseline_sft_0801.yaml";COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT required}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}";START="$SECONDS";STATUS=failed;RESULT=not_started;LOCAL=""
finish(){ local rc="$?";trap - EXIT;set +e;[[ "$rc" -eq 0 ]]&&STATUS=success;local elapsed="$((SECONDS-START))";printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\nresult=%s\n' "$STATUS" "$rc" "$elapsed" "$RESULT" >"$RUN_DIR/logs/status-${JOB_ID}.txt";QQ_SUB="Qwen3-VL v12.1 baseline SFT" QQ_BODY="status=${STATUS}; result=${RESULT}; job=${JOB_ID}; elapsed=${elapsed}s; output=${RUN_DIR}" "$WORKSPACE/.local/bin/wangyihan-send-qq" >"/tmp/v121_sft_${JOB_ID}.qq.log" 2>&1||true;rm -f "/tmp/v121_sft_${JOB_ID}.qq.log";[[ -z "$LOCAL" ]]||rm -rf -- "$LOCAL";exit "$rc";};trap finish EXIT
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$COMMIT" && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]||{ echo "code drift" >&2;exit 2; }
[[ "$(git -C "$LLAMA_FACTORY_ROOT" rev-parse HEAD)" == "$LLAMA_COMMIT" ]]||{ echo "LLaMA-Factory drift" >&2;exit 2; }
[[ -f "$DATASET_DIR/.data_complete" ]]||{ echo "data incomplete" >&2;exit 2; }
export PYTHONPATH="$LLAMA_FACTORY_ROOT/src:$ROOT" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
"$PYTHON" - "$DATASET_DIR" <<'PY'
import json,sys
p=sys.argv[1];v=json.load(open(p+'/validation.json'));assert v['valid'] is True and v['train_rows'] > 0 and v['val_rows'] > 0
rows=json.load(open(p+'/train.json'));assert len(rows)==v['train_rows'] and all(len(x['images'])==x['conversations'][0]['value'].count('<image>') for x in rows)
print('data_contract=ok train={} val={} round={}'.format(v['train_rows'],v['val_rows'],v.get('data_round')))
PY
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==2, torch.cuda.device_count()
assert all('A800' in torch.cuda.get_device_name(i) for i in range(2));assert torch.cuda.is_bf16_supported()
print('gpu_contract=ok devices='+','.join(torch.cuda.get_device_name(i) for i in range(2)))
PY
TMP="${SLURM_TMPDIR:-/tmp}";LOCAL="$(mktemp -d "$TMP/v121_sft_${JOB_ID}.XXXX")";OUT="$LOCAL/adapter"
printf '%s\n' "$COMMIT" >"$RUN_DIR/code_commit.txt";printf '%s\n' "$LLAMA_COMMIT" >"$RUN_DIR/llamafactory_commit.txt"
TRAIN_ROWS="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["train_rows"])' "$DATASET_DIR/validation.json")"
LEARNING_RATE="${LEARNING_RATE:-3.0e-5}";NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
"$PYTHON" - "$CONFIG" "$RUN_DIR/config_used.yaml" "$OUT" "$BASE_MODEL" "$DATASET_DIR" "$TRAIN_ROWS" "$LEARNING_RATE" "$NUM_TRAIN_EPOCHS" <<'PY'
from pathlib import Path
import json,sys
source,out,output_dir,model,dataset,max_samples,learning_rate,epochs=sys.argv[1:]
overrides={
    "output_dir": output_dir,
    "model_name_or_path": model,
    "dataset_dir": dataset,
    "media_dir": dataset,
    "max_samples": int(max_samples),
    "learning_rate": float(learning_rate),
    "num_train_epochs": float(epochs),
}
lines=[];seen=set()
for line in Path(source).read_text().splitlines():
    key=line.split(":",1)[0].strip() if ":" in line and not line.lstrip().startswith("#") else ""
    if key in overrides:
        value=overrides[key]
        rendered=json.dumps(value) if isinstance(value,str) else str(value)
        lines.append(f"{key}: {rendered}");seen.add(key)
    else:
        lines.append(line)
for key,value in overrides.items():
    if key not in seen:
        rendered=json.dumps(value) if isinstance(value,str) else str(value)
        lines.append(f"{key}: {rendered}")
Path(out).write_text("\n".join(lines)+"\n")
PY
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 "$LLAMA_FACTORY_ROOT/src/llamafactory/launcher.py" "$CONFIG" "output_dir=$OUT" "model_name_or_path=$BASE_MODEL" "dataset_dir=$DATASET_DIR" "media_dir=$DATASET_DIR" "max_samples=$TRAIN_ROWS" "learning_rate=$LEARNING_RATE" "num_train_epochs=$NUM_TRAIN_EPOCHS" 2>&1 | tee "$RUN_DIR/logs/train-${JOB_ID}.log"
[[ -f "$OUT/adapter_model.safetensors" && -f "$OUT/adapter_config.json" ]]||{ echo "adapter missing" >&2;exit 1; }
mkdir "$RUN_DIR/adapter";cp -a "$OUT/." "$RUN_DIR/adapter/"
"$PYTHON" - "$BASE_MODEL" "$RUN_DIR/adapter" "$RUN_DIR/merged_model" <<'PY'
import pathlib,sys,torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from peft import PeftModel
base,adapter,out=sys.argv[1:];model=Qwen3VLForConditionalGeneration.from_pretrained(base,dtype=torch.bfloat16,device_map='cpu')
model=PeftModel.from_pretrained(model,adapter).merge_and_unload();model.save_pretrained(out,safe_serialization=True,max_shard_size='5GB');AutoProcessor.from_pretrained(base).save_pretrained(out)
assert (pathlib.Path(out)/'model.safetensors').is_file()
PY
cp "$OUT/train_results.json" "$RUN_DIR/";cp "$OUT/trainer_state.json" "$RUN_DIR/" 2>/dev/null||true
"$PYTHON" - "$RUN_DIR" "$DATASET_DIR" <<'PY'
import hashlib,json,pathlib,sys
r,d=map(pathlib.Path,sys.argv[1:]);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();t=json.load(open(r/'train_results.json'))
m={'status':'success','train_loss':t.get('train_loss'),'train_runtime':t.get('train_runtime'),'dataset_manifest_sha256':h(d/'manifest.json'),'adapter_sha256':h(r/'adapter/adapter_model.safetensors'),'merged_model_sha256':h(r/'merged_model/model.safetensors')}
(r/'training_summary.json').write_text(json.dumps(m,indent=2)+'\n');(r/'.training_complete').write_text('complete\n');print(json.dumps(m))
PY
RESULT="training_complete;merged_model=$RUN_DIR/merged_model"
