import sys
import subprocess
import os

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: win_log_wrapper.py <stdout_file> <stderr_file> <command...>")
        sys.exit(1)

    stdout_file = sys.argv[1]
    stderr_file = sys.argv[2]
    cmd = sys.argv[3:]
    
    # Ensure log directories exist
    os.makedirs(os.path.dirname(stdout_file), exist_ok=True)
    os.makedirs(os.path.dirname(stderr_file), exist_ok=True)
    
    with open(stdout_file, "a", encoding="utf-8") as out_f, \
         open(stderr_file, "a", encoding="utf-8") as err_f:
        try:
            p = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
            p.wait()
            sys.exit(p.returncode)
        except Exception as e:
            err_f.write(f"win_log_wrapper failed to spawn: {e}\n")
            sys.exit(1)
