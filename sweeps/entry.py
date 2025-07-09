# sweeps/entry.py

import sys
import subprocess

def main():
    # 1. W&B agent가 실행한 커맨드 라인 인자를 리스트로 받습니다.
    # 예: wandb agent ... -> python sweeps/entry.py --compress=scc --optimizer=adamw
    # sys.argv[1:]는 ['--compress=scc', '--optimizer=adamw']가 됩니다.
    args = sys.argv[1:]
    
    hydra_overrides = []
    
    # 2. 리스트의 각 항목('--compress=scc' 등)에 대해 반복합니다.
    for arg in args:
        # 3. 각 항목의 맨 앞에 있는 '--'를 제거합니다.
        # 예: '--compress=scc'  ->  'compress=scc'
        clean_arg = arg.lstrip('-')
        hydra_overrides.append(clean_arg)

    # 4. 최종적으로 Hydra가 이해할 수 있는 커맨드 리스트를 만듭니다.
    # 결과: ['python', 'main.py', 'compress=scc', 'optimizer=adamw']
    cmd = ["python", "main.py"] + hydra_overrides
    
    print(f"🚀 Executing command: {' '.join(cmd)}")
    
    # 5. 완성된 커맨드로 main.py를 실행합니다.
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ main.py 실행 중 에러 발생: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()