#!/usr/bin/env python3
"""Small, guarded wrapper around Vast.ai's official `vastai` CLI."""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".env"
LOCAL_CLI = ROOT / "envs/vast-control/bin/vastai"


def env_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def config(args: argparse.Namespace) -> tuple[dict[str, str], Path | None, Path | None]:
    env_path = args.env_file.expanduser().resolve()
    key = os.environ.get("VAST_API_KEY") or env_value(env_path, "VAST_API_KEY")
    if not key:
        raise ValueError(f"Set VAST_API_KEY in {env_path} or the process environment")
    env = os.environ.copy()
    env["VAST_API_KEY"] = key
    private_name = args.private_key or os.environ.get("VAST_SSH_PRIVATE_KEY") or env_value(env_path, "VAST_SSH_PRIVATE_KEY")
    public_name = args.public_key or os.environ.get("VAST_SSH_PUBLIC_KEY") or env_value(env_path, "VAST_SSH_PUBLIC_KEY")
    private = Path(private_name).expanduser().resolve() if private_name else None
    public = Path(public_name).expanduser().resolve() if public_name else (Path(str(private) + ".pub") if private else None)
    return env, private, public


def vast(env: dict[str, str], *parts: str, raw: bool = True):
    command = [os.environ.get("VASTAI_BIN") or (str(LOCAL_CLI) if LOCAL_CLI.is_file() else "vastai"), *parts]
    if raw:
        command.append("--raw")
    try:
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Install the official Vast.ai CLI (`pip install vastai`)") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip().replace(env["VAST_API_KEY"], "[redacted]")
        raise RuntimeError(f"vastai {' '.join(parts[:2])} failed: {detail}") from exc
    if not raw:
        return result.stdout.strip()
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Vast CLI did not return JSON with --raw") from exc
    if isinstance(data, dict) and data.get("success") is False:
        raise RuntimeError(f"Vast API rejected request: {data.get('msg') or data.get('error') or 'unknown error'}")
    return data


def positive(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def instance_id(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive ID")
    return number


def find_offers(env: dict[str, str], query: str, disk: float, limit: int) -> list[dict]:
    rows = vast(env, "search", "offers", query, "--type", "on-demand", "--storage", str(disk),
                "--order", "dph_total", "--limit", str(limit))
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected Vast offer response")
    return rows


def price(offer: dict) -> float:
    value = offer.get("dph_total")
    if value is None:
        raise RuntimeError(f"Offer {offer.get('id')} has no total hourly price")
    return float(value)


def gib_from_mb(value) -> str:
    return f"{float(value) / 1024:.1f}" if value is not None else "?"


def unusual_memory(offer: dict) -> bool:
    name = str(offer.get("gpu_name", "")).upper().replace(" ", "")
    memory = float(offer.get("gpu_ram") or 0)
    return (name == "RTX4090" and memory > 26000) or (name == "CMP170HX" and memory > 12000)


def show_offers(rows: list[dict], max_hourly: float | None, limit: int) -> None:
    rows = [row for row in rows if max_hourly is None or price(row) <= max_hourly]
    rows.sort(key=price)
    print("offer_id  GPU × count              VRAM GiB  RAM GiB  disk GB  down Mb/s  reliability  $/hour  location")
    for row in rows[:limit]:
        model = str(row.get("gpu_name", "?")) + ("*" if unusual_memory(row) else "")
        print(f"{row.get('id', '?'):>8}  {model[:19]:19} ×{row.get('num_gpus', '?')!s:<2} "
              f"{gib_from_mb(row.get('gpu_ram')):>8}  {gib_from_mb(row.get('cpu_ram')):>7}  "
              f"{float(row.get('disk_space', 0)):>7.1f}  {float(row.get('inet_down', 0)):>9.0f}  "
              f"{float(row.get('reliability', 0)) * 100:>10.1f}%  "
              f"{price(row):>7.3f}  {row.get('geolocation', '?')}")
    if not rows:
        print("No matching on-demand offers.")
    if any(unusual_memory(row) for row in rows[:limit]):
        print("* Unusual advertised VRAM for this model; verify actual capacity and compute behavior before use.")
    print("Prices are live estimates for the requested disk size; transfer charges may be additional.")


def require_ssh_key(private: Path | None, public: Path | None) -> None:
    if not private or not private.is_file() or not public or not public.is_file():
        raise ValueError("Set VAST_SSH_PRIVATE_KEY and VAST_SSH_PUBLIC_KEY to existing local files")
    first = public.read_text().strip().split()
    if len(first) < 2 or not first[0].startswith(("ssh-", "ecdsa-")):
        raise ValueError(f"Invalid SSH public key: {public}")


def owned_instance(env: dict[str, str], number: int) -> dict:
    instance = vast(env, "show", "instance", str(number))
    if not isinstance(instance, dict) or not instance.get("id"):
        raise RuntimeError(f"Instance {number} was not found in your account")
    return instance


def ssh_target(env: dict[str, str], number: int) -> tuple[str, str, int]:
    value = vast(env, "ssh-url", str(number), raw=False)
    match = re.search(r"ssh://[^\s]+", value)
    if not match:
        raise RuntimeError(f"No SSH URL is available yet for instance {number}")
    url = urlsplit(match.group(0))
    if not url.hostname or not url.username or not url.port:
        raise RuntimeError("Vast returned an incomplete SSH URL")
    return url.username, url.hostname, url.port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--private-key", help="Local SSH private key path (overrides .env)")
    parser.add_argument("--public-key", help="Matching SSH public key path (overrides .env)")
    commands = parser.add_subparsers(dest="command", required=True)
    offers = commands.add_parser("offers", help="List on-demand offers, cheapest first")
    offers.add_argument("--min-vram", type=positive, required=True, help="GB per GPU, as reported by Vast")
    offers.add_argument("--gpus", type=instance_id, default=1)
    offers.add_argument("--disk-gb", type=positive, required=True)
    offers.add_argument("--min-ram", type=positive)
    offers.add_argument("--min-download-mbps", type=positive, help="Require reported download speed above this threshold")
    offers.add_argument("--min-reliability", type=positive, default=0.98)
    offers.add_argument("--gpu-name", help="Exact Vast GPU name, spaces allowed")
    offers.add_argument("--max-hourly", type=positive)
    offers.add_argument("--limit", type=instance_id, default=10)
    provision = commands.add_parser("provision", help="Create one confirmed on-demand instance")
    provision.add_argument("--offer-id", type=instance_id, required=True)
    provision.add_argument("--confirm-offer-id", type=instance_id, required=True)
    provision.add_argument("--disk-gb", type=positive, required=True)
    provision.add_argument("--max-hourly", type=positive, required=True)
    provision.add_argument("--min-vram", type=positive)
    provision.add_argument("--min-download-mbps", type=positive)
    provision.add_argument("--image", default="vastai/base-image:@vastai-automatic-tag")
    provision.add_argument("--label", default="rsrch_new")
    for name in ("status", "connect", "stop", "start", "terminate"):
        item = commands.add_parser(name)
        item.add_argument("instance_id", type=instance_id)
        if name == "terminate":
            item.add_argument("--confirm-instance-id", type=instance_id, required=True)
        if name == "connect":
            item.add_argument("--check", action="store_true", help="Verify SSH without opening a shell")
    args = parser.parse_args()
    env, private, public = config(args)
    if args.command == "offers":
        if args.min_reliability > 1:
            raise ValueError("Reliability must be between 0 and 1")
        query = (f"gpu_ram>={args.min_vram} num_gpus={args.gpus} "
                 f"disk_space>={args.disk_gb} reliability>={args.min_reliability} "
                 "direct_port_count>=1 verified=true rentable=true")
        if args.min_ram:
            query += f" cpu_ram>={args.min_ram}"
        if args.min_download_mbps:
            query += f" inet_down>{args.min_download_mbps}"
        if args.gpu_name:
            query += f" gpu_name={args.gpu_name.replace(' ', '_')}"
        show_offers(find_offers(env, query, args.disk_gb, max(50, args.limit * 5)), args.max_hourly, args.limit)
    elif args.command == "provision":
        if args.offer_id != args.confirm_offer_id:
            raise ValueError("Confirmed offer ID does not match")
        require_ssh_key(private, public)
        query = f"rentable=true verified=true disk_space>={args.disk_gb} direct_port_count>=1"
        if args.min_vram:
            query += f" gpu_ram>={args.min_vram}"
        if args.min_download_mbps:
            query += f" inet_down>{args.min_download_mbps}"
        # Vast's offer-ID query can return no rows even while the offer appears in
        # ordinary searches. Match the exact ID from a fresh price-sorted search.
        matches = find_offers(env, query, args.disk_gb, 1000)
        offer = next((row for row in matches if row.get("id") == args.offer_id), None)
        if offer is None:
            raise RuntimeError("Offer is no longer available; search again")
        if float(offer.get("disk_space", 0)) < args.disk_gb or int(offer.get("direct_port_count", 0)) < 1:
            raise RuntimeError("Offer no longer has the requested disk or direct SSH port")
        if args.min_vram and float(offer.get("gpu_ram", 0)) < args.min_vram * 1000:
            raise RuntimeError("Offer no longer meets the VRAM requirement")
        if args.min_download_mbps and float(offer.get("inet_down", 0)) <= args.min_download_mbps:
            raise RuntimeError("Offer no longer meets the download-speed requirement")
        current_price = price(offer)
        if current_price > args.max_hourly:
            raise RuntimeError(f"Current price ${current_price:.3f}/hour exceeds ${args.max_hourly:.3f}/hour")
        created = vast(env, "create", "instance", str(args.offer_id), "--image", args.image,
                       "--disk", str(args.disk_gb), "--ssh", "--direct", "--label", args.label,
                       "--cancel-unavail")
        number = created.get("new_contract") if isinstance(created, dict) else None
        if not isinstance(number, int):
            raise RuntimeError(f"Create response has no instance ID: {created}")
        print(f"Created instance {number} from offer {args.offer_id} at estimated ${current_price:.3f}/hour plus any transfer charges.")
        try:
            attached = vast(env, "attach", "ssh", str(number), str(public), raw=False)
            response = ast.literal_eval(attached)
            if not isinstance(response, dict) or response.get("success") is not True:
                raise RuntimeError(f"Vast rejected SSH key attachment: {response}")
        except Exception:
            print(f"SSH key attachment failed; instance {number} still exists and may be billing. Check status or terminate it.", file=sys.stderr)
            raise
        print(f"SSH public key attached. Run `python3 scripts/vast_gpu.py status {number}`, then `connect {number} --check`.")
    elif args.command == "status":
        instance = owned_instance(env, args.instance_id)
        fields = ("id", "actual_status", "intended_status", "gpu_name", "num_gpus", "gpu_ram", "dph_total", "disk_space", "status_msg")
        print(json.dumps({field: instance.get(field) for field in fields}, indent=2))
    elif args.command == "connect":
        require_ssh_key(private, public)
        instance = owned_instance(env, args.instance_id)
        if instance.get("actual_status") != "running":
            raise RuntimeError(f"Instance is {instance.get('actual_status')}; SSH is not ready")
        user, host, port = ssh_target(env, args.instance_id)
        command = ["ssh", "-i", str(private), "-p", str(port), "-o", "IdentitiesOnly=yes",
                   "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10", f"{user}@{host}"]
        if args.check:
            subprocess.run(command[:-1] + ["-o", "BatchMode=yes", command[-1], "true"], check=True)
            print(f"SSH verified for instance {args.instance_id}: {user}@{host}:{port}")
        else:
            os.execvp("ssh", command)
    else:
        if args.command == "terminate" and args.confirm_instance_id != args.instance_id:
            raise ValueError("Confirmed instance ID does not match")
        owned_instance(env, args.instance_id)
        action = "destroy" if args.command == "terminate" else args.command
        outcome = vast(env, action, "instance", str(args.instance_id),
                       *(["-y"] if action == "destroy" else []), raw=False)
        if not outcome.startswith({"stop": "stopping instance", "start": "starting instance",
                                   "destroy": "destroying instance"}[action]):
            raise RuntimeError(f"Vast did not confirm the {action} request: {outcome}")
        print(f"{args.command.capitalize()} requested for instance {args.instance_id}.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
