# VAMOS 데이터셋을 이용한 OpenVLA LoRA 학습 가이드

## 개요
VAMOS navigation 데이터셋을 사용하여 OpenVLA 모델을 2D 속도(v_x, v_y) 예측에 특화하도록 LoRA fine-tuning하는 방법입니다.

## 구현된 내용

### 1. 커스텀 데이터셋 클래스
- **파일**: `prismatic/vla/datasets/vamos_dataset.py`
- **기능**:
  - Parquet 파일 형식의 VAMOS 데이터셋 로딩
  - `trajectory_3d`에서 2D 속도(v_x, v_y) 계산
  - 이미지 및 언어 지시사항 처리
  - 데이터셋 통계 자동 계산 (정규화용)

### 2. Fine-tuning 스크립트 수정
- **파일**: `vla-scripts/finetune.py`
- **변경사항**:
  - `dataset_name == "vamos"`일 때 VamosDataset 자동 사용
  - 2D 속도에 맞는 액션 토크나이저 설정 (min_action=-1, max_action=1)
  - PyTorch DataLoader 설정 (shuffle, num_workers 등)

### 3. 액션 차원
- **기존**: 7D (EEF delta + gripper)
- **변경**: 2D (v_x, v_y만)
- 모델은 데이터셋 통계에서 자동으로 액션 차원을 인식합니다.

## 실행 방법

### 기본 명령어

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/jydisk/dataset/vamos_dataset \
  --dataset_name vamos \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps 5000 \
  --max_steps 200000
```

### 주요 파라미터 설명

- `--data_root_dir`: VAMOS 데이터셋이 있는 디렉토리 (예: `/mnt/jydisk/dataset/vamos_dataset`)
- `--dataset_name`: 반드시 `vamos`로 설정
- `--batch_size`: GPU 메모리에 따라 조정 (A100 80GB 기준 16-24)
- `--lora_rank`: LoRA rank (기본값: 32)
- `--image_aug`: 이미지 augmentation 사용 여부 (False 권장)
- `--max_steps`: 최대 학습 스텝 수

### GPU 메모리 최적화

GPU 메모리가 부족한 경우:
- `--batch_size`를 줄이고 `--grad_accumulation_steps`를 늘려서 effective batch size 유지
- 예: `--batch_size 8 --grad_accumulation_steps 2` (effective batch size = 16)

### 멀티 GPU 학습

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <NUM_GPUS> vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/jydisk/dataset/vamos_dataset \
  --dataset_name vamos \
  ...
```

## 데이터셋 구조

VAMOS 데이터셋은 다음 구조를 가정합니다:
```
vamos_dataset/
  data/
    train-*.parquet
    validation-*.parquet
```

각 parquet 파일은 다음 컬럼을 포함합니다:
- `trajectory_3d`: 3D 좌표 배열 (속도 계산용)
- `image`: JPEG 이미지 bytes
- `lang_goal` 또는 `text`: 언어 지시사항

## 학습 결과

학습이 완료되면 다음 파일들이 생성됩니다:
- `run_root_dir/<exp_id>/dataset_statistics.json`: 데이터셋 통계 (inference용)
- `run_root_dir/<exp_id>/`: 최종 모델 체크포인트
- `adapter_tmp_dir/<exp_id>/`: LoRA 어댑터 가중치 (임시)

## Inference 사용법

학습된 모델을 사용하여 2D 속도를 예측:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# 모델 로드
processor = AutoProcessor.from_pretrained("<CHECKPOINT_DIR>", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "<CHECKPOINT_DIR>",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda:0")

# 이미지와 지시사항 준비
image = Image.open("path/to/image.jpg")
instruction = "navigate to goal"

# 액션 예측 (v_x, v_y 반환)
prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="vamos_dataset", do_sample=False)

print(f"Predicted velocity: v_x={action[0]:.4f}, v_y={action[1]:.4f}")
```

## 주의사항

1. **액션 범위**: 학습 시 액션 토크나이저는 [-1, 1] 범위를 사용합니다. 실제 속도 값은 데이터셋 통계를 사용하여 denormalize됩니다.

2. **데이터셋 통계**: `dataset_statistics.json` 파일이 inference에 필요합니다. 이 파일이 없으면 모델이 액션 차원을 인식하지 못할 수 있습니다.

3. **이미지**: 현재 구현은 각 trajectory의 첫 번째 이미지만 사용합니다. 각 timestep마다 다른 이미지를 사용하려면 데이터셋 클래스를 수정해야 합니다.

## 문제 해결

### 메모리 부족 오류
- `--batch_size`를 줄이거나 `--grad_accumulation_steps`를 늘립니다.
- `--use_quantization True`를 사용하여 4-bit quantization을 활성화할 수 있습니다 (성능 저하 가능).

### 데이터셋 로딩 오류
- `data_root_dir` 경로가 올바른지 확인하세요.
- `data/` 서브디렉토리에 parquet 파일이 있는지 확인하세요.

### 액션 차원 오류
- `dataset_statistics.json` 파일이 올바르게 생성되었는지 확인하세요.
- 통계 파일의 `action` 필드가 2D 배열인지 확인하세요.


