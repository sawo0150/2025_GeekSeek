# sweeps/varnet/train_entry.py

import argparse, subprocess, sys, os

def parse_args():
    """
    W&B의 command 섹션으로부터 --key=value 형태의 인수를 받도록 정의합니다.
    """
    parser = argparse.ArgumentParser()
    
    # command가 전달하는 모든 인수를 여기에 정의합니다.
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--config-name", type=str, required=False)
    
    # 나중에 sweep할 다른 파라미터가 있다면 여기에 추가하면 됩니다.
    # 예: parser.add_argument("--optimizer", type=str)

    return parser.parse_known_args()

# 1. W&B agent의 `command:`로부터 인수를 파싱합니다.
#    (예: --config-name=..., --model=...)
args, unknown = parse_args()

# 2. 최종적으로 Hydra(main.py)에 전달할 명령어를 새로 생성합니다.
cmd = ["python", "main.py"]

# 3. 파싱된 인수를 Hydra가 이해하는 형식으로 "번역" 및 "재구성"합니다.

# (A) --config-name은 Hydra 특별 명령이므로 "--"를 유지한 채 그대로 전달합니다.
if args.config_name:
    cmd.append(f"--config-name={args.config_name}")

# (B) model 인수는 Hydra 값 오버라이드이므로 "--"를 제거하고 "key=value"로 만듭니다.
#     선생님의 다른 핵심 로직(exp_name 설정 등)도 여기에 포함됩니다.
if args.model:
    cmd.append(f"model={args.model}")
    cmd.append(f"exp_name=varnet_sweep_{args.model}")

# (C) 만약 maskDuplicate나 scheduler 등 다른 로직이 필요했다면 여기에 추가되었을 것입니다.
#     (이번 sweep에는 model만 있으므로 생략)

# (D) 혹시 모를 나머지 인수를 처리합니다.
for arg in unknown:
    if arg.startswith('--') and '=' in arg:
        cmd.append(arg[2:])
    else:
        cmd.append(arg)
        
# 4. 최종 생성된, Hydra가 완벽하게 이해할 수 있는 명령어를 실행합니다.
print(f"Executing command: {' '.join(cmd)}")
sys.exit(subprocess.call(cmd))