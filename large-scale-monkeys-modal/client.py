from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "openai==1.38.0", "datasets==2.20.0"
)

app = modal.App("eval-infinite-monkeys", image=image)

volume = modal.Volume.from_name("humaneval", create_if_missing=True)
DATA_DIR = Path("/mnt/humaneval")

default_system_prompt = "Write the body for the Python function provided in the prompt below. Do not write anything else. Your output will be directly concatenated with the prompt and the resulting function executed against tests."

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@dataclass
class CompletionParams:
    model: str = None
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0
    presence_penalty: float = 0
    n: int = 1
    stop: str = None
    seed: int = None


@dataclass
class ClientParams:
    app_name: str = "example-infinite-monkeys"
    workspace: str = None
    api_key: str = "super-secret-token"

    @property
    def url(self):
        return f"https://{self.workspace}--{self.app_name}-serve.modal.run/v1"


@app.local_entrypoint()
def main(
    app_name: str = "example-infinite-monkeys",
    workspace: str = None,
    api_key: str = "super-secret-token",
    model: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    n: int = 1,
    stop: str = None,
    seed: int = None,
    data_dir: str = "dev-llm",
    subsample: int = 1,
    system_prompt: str = default_system_prompt,
    dry_run: bool = True,
):
    if workspace is None:
        workspace = modal.config._profile

    client_params = ClientParams(app_name, workspace, api_key)

    completion_params = CompletionParams(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=n,
        stop=stop,
        seed=seed,
    )

    save_dataset.remote(path=data_dir, subsample=subsample)

    results = run_human_eval.remote(
        client_params=client_params,
        completion_params=completion_params,
        system_prompt=system_prompt,
        data_dir=data_dir,
        dry_run=dry_run,
    )
    if results:
        with open("/tmp/results.jsonl", "w") as f:
            f.writelines(json.dumps(result) + "\n" for result in results)
        print(f"results saved locally to {f.name}")


@app.function(volumes={DATA_DIR: volume}, timeout=1 * HOURS)
def run_human_eval(
    client_params: ClientParams,
    completion_params: CompletionParams,
    data_dir="dev-llm",
    system_prompt: str = default_system_prompt,
    dry_run=True,
):
    dataset = load_dataset(data_dir)

    timestamp = datetime.utcnow().isoformat() + "Z"
    output_dir = Path(DATA_DIR) / data_dir / f"run-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    for i, item in enumerate(dataset):
        handles.append(
            run_item.spawn(
                item,
                client_params,
                completion_params,
                system_prompt,
                output_dir,
                dry_run,
            )
        )

    for handle in handles:
        result = handle.get()

    if not dry_run:
        return result


@app.function(volumes={DATA_DIR: volume}, timeout=1 * HOURS)
def run_item(
    item: dict,
    client_params: ClientParams,
    completion_params: CompletionParams,
    system_prompt: str,
    output_dir: Path,
    dry_run: bool,
):
    client = create_client(client_params)
    if completion_params.model:
        print(
            Colors.BOLD,
            f"🧠: Using model {completion_params.model}. This may trigger a model load on first call!",
            Colors.END,
            sep="",
        )
    else:
        print(
            Colors.BOLD,
            f"🔎: Looking up available models on server at {client.base_url}. This may trigger a model load!",
            Colors.END,
            sep="",
        )
        model = client.models.list().data[0]
        model = model.id
        print(
            Colors.BOLD,
            f"🧠: Using {model}",
            Colors.END,
            sep="",
        )
        completion_params.model = model

    prompt = item["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    per_request = 250
    ct, completions = completion_params.n, []
    if not dry_run:
        while ct > 0:
            response = get_completion(
                client,
                messages=messages,
                **asdict(completion_params) | dict(n=min(ct, per_request)),
            )
            if response:
                completions += [
                    {
                        "task_id": item["task_id"],
                        "completion": choice.message.content,
                    }
                    for choice in response.choices
                ]
            ct -= per_request

        index = item["task_id"].split("/")[-1]
        output_path = output_dir / f"{index}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(json.dumps(completion) + "\n" for completion in completions)

        print(Colors.GREEN + f"Completions saved to {output_path}" + Colors.END)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


def get_completion(client, **kwargs):
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(Colors.RED, f"Error during API call: {e}", Colors.END, sep="")
        return None


def create_client(client_params: ClientParams):
    from openai import OpenAI

    client = OpenAI(api_key=client_params.api_key)
    client.base_url = client_params.url

    return client


@app.function(volumes={DATA_DIR: volume})
def save_dataset(path="dev-llm", subsample: int = 1):
    import datasets

    path = DATA_DIR / path

    ds = datasets.load_dataset(
        "openai/openai_humaneval",
        split=datasets.ReadInstruction("test", to=subsample, unit="%"),
    )

    ds.to_json(path / "data.jsonl")

    volume.commit()


def load_dataset(path="dev-llm"):
    import datasets

    path = DATA_DIR / path

    ds = datasets.load_dataset(path=str(path), data_files="data.jsonl")

    return ds["train"]
