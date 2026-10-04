# HSE cluster training and Slurm jobs

For training on `cluster` / `clust`, preparing sbatch files, quick GPU tests,
resource availability, job status, cancellation, or checkpoint resume, use the
installed [hse-cluster skill](/mnt/c/Users/ogure/.codex/skills/hse-cluster/SKILL.md).
It reuses the existing `redshield-vpn` connection skill and includes a batch
template, scheduler commands, and live-verified queue limits. Recheck account
and queue limits before submitting; creating the skill did not submit training.

# Vast.ai GPU lifecycle

Use this project workflow when the user asks to find, rent, connect to, stop, restart, or terminate a Vast.ai GPU. The implementation is [`scripts/vast_gpu.py`](scripts/vast_gpu.py), a guarded wrapper around the [official Vast.ai CLI](https://github.com/vast-ai/vast-cli). Version 1.8.2 is installed in ignored `envs/vast-control/`, which the script finds automatically. To recreate it, run `python3 -m venv envs/vast-control && envs/vast-control/bin/python -m pip install vastai==1.8.2`. Keep it separate from the Qwen and FLUX training environments.

## Local credentials

Put these values in the ignored `/home/kolyangg/rsrch_new/.env` (see `.env.example`):

```dotenv
VAST_API_KEY=your_vast_api_key
VAST_SSH_PRIVATE_KEY=/absolute/local/path/to/private_key
VAST_SSH_PUBLIC_KEY=/absolute/local/path/to/private_key.pub
```

Get the API key from [Vast.ai Manage Keys](https://console.vast.ai/manage-keys/). The private key stays on this machine; the script attaches only its public key to the instance. Never print the API key or private key, put either in a command argument, upload them to a rented host, or commit `.env`. The CLI reads `VAST_API_KEY` from its child process environment. The script also accepts `--env-file`, `--private-key`, and `--public-key` before the subcommand.

## Find and present offers

Translate the user's requested GPU count, per-GPU VRAM, disk, RAM, reliability, exact GPU model, and hourly budget into a search. Ask for material missing rental parameters if they cannot be inferred; a 48 GB/80 GB research profile is not itself authorization to rent or run training. Default search is verified, rentable, on-demand, one GPU, reliability at least 0.98, and a direct SSH port. Examples:

```bash
python3 scripts/vast_gpu.py offers --min-vram 48 --disk-gb 250 --min-ram 64 --max-hourly 2.00
python3 scripts/vast_gpu.py offers --min-vram 80 --disk-gb 500 --gpu-name H100_SXM --limit 8
python3 scripts/vast_gpu.py offers --min-vram 48 --disk-gb 200 --min-download-mbps 1000 --min-reliability 0.01
```

Present several live options with offer ID, GPU model/count/VRAM, RAM, disk, reliability, location, and estimated USD/hour for the requested disk. Mention possible transfer charges. Offers and prices change; recheck the selected offer before provision. State the exact image, disk size, hourly ceiling, and SSH mode to be used. Wait for the user to confirm one specific offer and configuration before creating an instance. Do not silently select the cheapest or use interruptible/bid pricing.

Vast's query parser accepts `gpu_ram` and `cpu_ram` thresholds in GB but `--raw` returns those fields in MB; the wrapper converts them to GiB for display and checks the raw units when provisioning. Treat unusual reported memory on CMP 170HX and RTX 4090 listings as unverified hardware configurations. Verify actual VRAM and usable compute on a rented host before starting this research workload.

## Provision and connect

After that confirmation, run the guarded command with the confirmed offer ID repeated and a maximum hourly price. The default image is `vastai/base-image:@vastai-automatic-tag`; use a user-approved alternative if needed. The script rechecks offer availability, disk and direct SSH port, and the current total hourly price before calling `create instance`. It creates one on-demand SSH instance, attaches the public key, and prints the new instance ID.

```bash
python3 scripts/vast_gpu.py provision --offer-id OFFER_ID --confirm-offer-id OFFER_ID \
  --disk-gb 250 --max-hourly 2.00 --min-vram 48 --min-download-mbps 1000
python3 scripts/vast_gpu.py status INSTANCE_ID
python3 scripts/vast_gpu.py connect INSTANCE_ID --check
python3 scripts/vast_gpu.py connect INSTANCE_ID
```

Poll `status` at reasonable intervals while the image loads. `connect --check` verifies the key and SSH route without opening a shell. If attachment or SSH fails after creation, report the instance ID immediately because billing may already have started; inspect status and repair the connection or terminate on the user's instruction. Do not submit training jobs as part of provisioning. For this research, follow [`docs/deployment.md`](docs/deployment.md) only after a suitable host and access path are established.

## Stop, restart, terminate

```bash
python3 scripts/vast_gpu.py stop INSTANCE_ID
python3 scripts/vast_gpu.py start INSTANCE_ID
python3 scripts/vast_gpu.py terminate INSTANCE_ID --confirm-instance-id INSTANCE_ID
```

Always inspect `status` and use the user's intended instance ID. Stop preserves instance data but disk billing continues, and restart depends on host capacity. Terminate destroys the instance and its local data. Before termination, check whether results, checkpoints, and logs need transfer. Report the action's actual API result and, when possible, follow with `status` to verify the state. Never describe a submitted stop/destroy request as fully complete until its resulting state is observed.

## API sources

This workflow uses the official CLI's [search offers](https://docs.vast.ai/cli/reference/search-offers), [create instance](https://docs.vast.ai/cli/reference/create-instance), [attach ssh](https://docs.vast.ai/cli/reference/attach-ssh), [ssh-url](https://docs.vast.ai/cli/reference/ssh-url), [show instance](https://docs.vast.ai/cli/reference/show-instance), [stop](https://docs.vast.ai/cli/reference/stop-instance), [start](https://docs.vast.ai/cli/reference/start-instance), and [destroy](https://docs.vast.ai/cli/reference/destroy-instance) APIs. The CLI's `--raw` output is used for machine-readable offer and instance responses.
