from pathlib import Path

import modal

app = modal.App("humaneval-sandbox")

volume = modal.Volume.from_name("humaneval", create_if_missing=True)

sandbox_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/modal-labs/human-eval.git",
        "pip install -e human-eval",
    )
)

MINUTES = 60


@app.function(volumes={"/humaneval": volume})
def run_humaneval(sample_file_path: str, problem_file_path: str):
    with modal.Volume.ephemeral() as vol:
        with vol.batch_upload() as batch:
            batch.put_file(sample_file_path, "samples.jsonl")
            batch.put_file(problem_file_path, "problems.jsonl")

        print(f"Starting sandbox for {sample_file_path}")
        sandbox = modal.Sandbox.create(
            "bash",
            "-c",
            "evaluate_functional_correctness vol/samples.jsonl --problem_file=vol/problems.jsonl --n_workers=32",
            image=sandbox_image,
            volumes={"/vol": vol},
            timeout=5 * MINUTES,
            cpu=32,
        )

        try:
            sandbox.wait_for(4 * MINUTES)
            print(f"Finished sandbox for {sample_file_path}")
        except TimeoutError:
            print("Sandbox timed out")

        if sandbox.returncode == 0:
            print(sandbox.stdout.read())
            data = b""
            for chunk in vol.read_file("samples.jsonl_results.jsonl"):
                data += chunk
            with open(f"{sample_file_path}_results.jsonl", "wb") as f:
                f.write(data)
        else:
            print(f"Tests failed with code {sandbox.returncode}")
            print(sandbox.stderr.read())


@app.function(volumes={"/humaneval": volume}, timeout=10 * MINUTES)
def find_missing_files():
    import os

    volume.reload()

    # Find all files matching /humaneval/{env}/{run}/{id}.jsonl
    envs = [element for element in Path("/humaneval").iterdir() if element.is_dir()]
    for env in envs:
        print(f"looking in {env}")
        problem_file = env / "data.jsonl"

        pattern = "*/*.jsonl"
        handles = []
        for file_path in env.glob(pattern):
            # Skip files that end with _results.jsonl
            if str(file_path).endswith("_results.jsonl"):
                continue

            print(f"Checking {file_path}")
            # Check if the corresponding results file exists
            results_file = f"{file_path}_results.jsonl"
            if not os.path.exists(results_file):
                # If it doesn't exist, run run_humaneval
                handles.append(run_humaneval.spawn(file_path, problem_file))

        for handle in handles:
            handle.get()


@app.local_entrypoint()
def main():
    find_missing_files.remote()
