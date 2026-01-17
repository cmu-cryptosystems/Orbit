import argparse
import subprocess
import threading
from queue import Queue
from typing import List


def worker(queue: Queue, base_cmd: List[str]):
    """Worker thread that processes runs from the queue."""
    while True:
        run_num = queue.get()
        if run_num is None:
            break
        
        cmd = base_cmd + ["--run", str(run_num)]
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed run {run_num}")
        except subprocess.CalledProcessError as e:
            print(f"Error running with --run {run_num}: {e}")
        finally:
            queue.task_done()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ResNet", choices=["ResNet", "AlexNet", "SqueezeNet", "MobileNet", "VGG16"])
    parser.add_argument("--act", default="SiLU", choices=["ReLU", "SiLU"])
    parser.add_argument("--n", type=int, default=16, choices=[16, 64])
    parser.add_argument("--Lm", type=int, default=16)
    parser.add_argument("--Sw", type=int, default=40)
    parser.add_argument("--Csw", type=int, default=None)
    parser.add_argument("--plain", action='store_true')
    parser.add_argument("--maxthread", type=int, default=8)
    
    args = parser.parse_args()
    
    # Build base command
    base_cmd = [
        "python", "-u", "run_one_test.py",
        "--model", args.model,
        "--act", args.act,
        "--n", str(args.n),
        "--Lm", str(args.Lm),
        "--Sw", str(args.Sw),
    ]
    if args.Csw is not None:
        base_cmd += ["--Csw", str(args.Csw)]
    if args.plain:
        base_cmd.append("--plain")
    
    # Create queue and threads
    queue = Queue()
    threads = []
    
    for _ in range(args.maxthread):
        t = threading.Thread(target=worker, args=(queue, base_cmd))
        t.start()
        threads.append(t)
    
    # Queue all runs
    for run_num in range(24):
        queue.put(run_num)
    
    # Wait for completion
    queue.join()
    
    # Stop workers
    for _ in range(args.maxthread):
        queue.put(None)
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()