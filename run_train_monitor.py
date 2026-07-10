import subprocess
import time
import sys

def main():
    print("🚀 Starting training process with human-readable categories...")
    # Start the training process
    process = subprocess.Popen(
        [sys.executable, "-u", "src/main.py"],
        env={"PYTHONPATH": "."},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    start_time = time.time()
    last_print = start_time

    # Read output line by line as it arrives
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            # Print the line from the training script
            print(line.strip(), flush=True)
            last_print = time.time()
        else:
            # Sleep briefly and print status if there is a long silence
            time.sleep(1)
            current_time = time.time()
            if current_time - last_print >= 15:
                elapsed = int(current_time - start_time)
                print(f"[Monitor] Still training... ({elapsed} seconds elapsed)", flush=True)
                last_print = current_time

    # Process finished
    return_code = process.poll()
    print(f"\n[Monitor] Training finished with exit code {return_code}")
    if return_code != 0:
        sys.exit(return_code)

if __name__ == "__main__":
    main()
