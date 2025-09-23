import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import timedelta

sns.set_theme(
    context="notebook",   # scale: "paper", "notebook", "talk", "poster"
    style="whitegrid",    # background: "white", "whitegrid", "dark", "darkgrid", "ticks"
    palette="deep"        # colors: "deep", "muted", "bright", "pastel", "dark", "colorblind"
)

plt.rcParams["figure.dpi"] = 150   # sharper figures
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.figsize"] = (12, 6) # Sets the default figure size to 12 inches wide and 6 inches tall.


# override for black/transparent background and neon green
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "gray",
    "axes.labelcolor": "gray",
    "xtick.color": "gray",
    "ytick.color": "gray",
    "grid.color": "gray",
    "text.color": "gray",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
})


line_re = re.compile(
    r"step:(\d+)/(\d+)"
    r"(?:\s+val_loss:([0-9.]+))?"
    r"\s+train_time:(\d+)ms"
    r"\s+step_avg:([0-9.]+)ms"
)



def plot_time_loss(df):
    
    sns.lineplot(data=df, x="train_time_ms", y="val_loss", marker="o", color="lime")

    for _, row in df.iterrows():
        plt.text(
            x=row["train_time_ms"],
            y=row["val_loss"],
            s=str(int(row["step"])),
            ha="center",
            va="bottom",
            color="white",
            fontsize=9
        )
    
    # Attain ≤3.28 mean val loss
    target_loss = 3.28
    
    plt.axhline(
        y=target_loss,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"target val_loss: {target_loss}"
    )
    
    
    best_record_ms = timedelta(minutes=2, seconds=52).total_seconds() * 1000
    
    plt.axvline(
        x=best_record_ms,
        color="orange",
        linestyle="--",
        linewidth=1.,
        label=f"best record: {humanize_ms(best_record_ms)}"
    )
    
    # plt.yscale("log")
    plt.title("Validation Loss vs Train Time")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: humanize_ms(x)))
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Elapsed Train Time')
    plt.legend()
    plt.show()

def parse_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            step, step_total, val_loss, train_time, step_avg = m.groups()
            rows.append({
                "step": int(step),
                "step_total": int(step_total),
                "val_loss": float(val_loss) if val_loss else None,
                "train_time_ms": int(train_time),
                "step_avg_ms": float(step_avg),
            })
    return pd.DataFrame(rows)

# Parse Results 

import re, os, glob, datetime
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from tqdm.auto import tqdm

STEP_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+train_time:(\d+)ms\s+step_avg:([0-9.]+)ms"
)
MEM_RE = re.compile(
    r"peak memory allocated:\s+(\d+)\s+MiB\s+reserved:\s+(\d+)\s+MiB"
)

def _tail_lines(path, n=80):
    with open(path, "rb") as f:
        try:
            f.seek(-4096, os.SEEK_END)
        except OSError:
            f.seek(0)
        data = f.read().decode("utf-8", errors="ignore")
    return data.splitlines()[-n:]

def parse_run_dir(path: str) -> str:
    try:
        # extract the last component, e.g. "071225_BosAlign"
        name = path.rstrip("/").split("/")[-1]
        date_part, title_part = name.split("_", 1)
        # YYMMDD → datetime
        dt = datetime.strptime(date_part, "%m%d%y")
        # format: "2025 Jul 07"
        date_str = dt.strftime("%Y %b %d")
        # insert space before capital letters in title part
        title_str = "".join(
            [" " + c if c.isupper() else c for c in title_part]
        ).strip()
        return f"{date_str} - {title_str}"
    except Exception:
        return path


def parse_date_from_label(s: str) -> pd.Timestamp:
    # supports "2025 Jul 07 - Bos Align" and "071225_BosAlign"
    s = s.strip()
    # format A: "YYYY Mon DD - ..."
    m = re.match(r"^(\d{4})\s+([A-Za-z]{3})\s+(\d{2})\b", s)
    if m:
        return pd.to_datetime(" ".join(m.groups()), format="%Y %b %d")
    # format B: "YYMMDD_..."
    m = re.match(r"^(\d{2})(\d{2})(\d{2})_", s)
    if m:
        mo, d, y = m.groups()
        return pd.to_datetime(f"20{y}-{mo}-{d}")
    raise ValueError(f"Unrecognized label date: {s}")



def humanize_ms(ms):
    try:
        seconds = ms / 1000
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"
        elif minutes:
            return f"{int(minutes)}m {sec:.1f}s"
        else:
            return f"{sec:.1f}s"
    except:
        return ms
        
def parse_last_metrics_from_log(path):
    last = _tail_lines(path)
    step_m = mem_m = None
    # scan from bottom up for robustness
    for line in reversed(last):
        if step_m is None:
            step_m = STEP_RE.search(line)
        if mem_m is None:
            mem_m  = MEM_RE.search(line)
        if step_m and mem_m:
            break
    if not step_m:
        return None
    step, step_total, val_loss, train_time_ms, step_avg_ms = step_m.groups()
    peak_mib = reserved_mib = None
    if mem_m:
        peak_mib, reserved_mib = mem_m.groups()
    st = os.stat(path)

    return {
        'date': parse_date_from_label(os.path.split(os.path.dirname(path))[-1]),
        "log_path": path,
        "run_label": os.path.split(os.path.dirname(path))[-1].split('_')[-1],
        "log_file": os.path.basename(path),
        "mtime": datetime.fromtimestamp(st.st_mtime),
        "step": int(step),
        "step_total": int(step_total),
        "val_loss": float(val_loss),
        "train_time_ms": int(train_time_ms),
        "step_avg_ms": float(step_avg_ms),
        "peak_mem_mib": int(peak_mib) if peak_mib else None,
        "reserved_mem_mib": int(reserved_mib) if reserved_mib else None,
    }

def collect_records(base="records", exclude='GPT2Medium'):
    paths = glob.glob(os.path.join(base, "**", "*.txt"), recursive=True)
    rows = []
    for p in tqdm(paths):
        if exclude in p:
            continue
        m = parse_last_metrics_from_log(p)
        if m:
            rows.append(m)
    if not rows:
        raise RuntimeError("No parsable logs found.")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # copy the last non-null downwards
    df['train_time_ms'] = df['train_time_ms'].ffill()
    df['count'] = 1

    return df



import os
import shutil
from datetime import datetime
import subprocess


def ensure_clean_repo():
    # check for uncommitted changes
    status = subprocess.run(["git", "status", "--porcelain"], 
                            capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError("Uncommitted changes present. Commit or stash before running.")

    # get current commit hash
    result = subprocess.run(["git", "rev-parse", "HEAD"], 
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()

if __name__ == "__main__":
    src_dir = "./logs"
    dst_root = "./myruns"

    git_hash = ensure_clean_repo()

    # date in DDMMYY format
    date_str = datetime.now().strftime("%m%d%y")
    dst_dir = os.path.join(dst_root, date_str)
    dst_dir = f'{dst_dir}_{git_hash}'
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if fname.endswith(".txt"):
            # embed git hash in filename
            base, ext = os.path.splitext(fname)
            new_name = f"{base}_{git_hash[:7]}{ext}"
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, new_name)
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} -> {dst_path}")

    my_records = collect_records("./myruns", exclude='GPT2Medium')    
    speedrun_records = collect_records("./records", exclude='GPT2Medium')
    # Add a marker to distinguish
    my_records["source"] = "mine"
    speedrun_records["source"] = "official"

    # Concatenate
    combined = pd.concat([my_records, speedrun_records], ignore_index=True)

    # Sort by val_loss ascending (better = lower loss)
    combined = combined.sort_values(by=["train_time_ms", "val_loss"]).reset_index(drop=True)

    # Add rank
    combined["rank"] = combined.index + 1

    # Extract only my runs with their rank
    headers = ["date","run_label", "val_loss", "train_time_ms", "rank"]
    my_ranks = combined[combined["source"] == "mine"][headers]
    
    print(my_ranks) 
    
    top = combined.sort_values(
        by=["train_time_ms", 'val_loss']
    ).head(1)

    print(top[headers])