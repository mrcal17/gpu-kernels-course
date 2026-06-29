"""Print the real, queried properties of your GPU.

Run:  python -m harness.device_info     (from the repo root)

Phase 0 leans on these numbers. Don't trust a blog's spec sheet when the
driver will tell you the truth for *your* card.
"""
import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available to PyTorch.")
        return

    idx = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(idx)

    print(f"Device              : {p.name}")
    print(f"Compute capability  : sm_{p.major}{p.minor}")
    print(f"SM count            : {p.multi_processor_count}")
    print(f"Total global memory : {p.total_memory / 1e9:.2f} GB")

    # These attribute names vary slightly across torch versions; fall back gracefully.
    interesting = [
        "warp_size",
        "max_threads_per_multi_processor",
        "max_threads_per_block",
        "regs_per_multiprocessor",
        "shared_memory_per_multiprocessor",
        "shared_memory_per_block",
        "shared_memory_per_block_optin",
        "L2_cache_size",
        "l2_cache_size",
        "memory_clock_rate",
        "memory_bus_width",
    ]
    print("\n-- selected limits --")
    for attr in interesting:
        if hasattr(p, attr):
            print(f"  {attr:34}: {getattr(p, attr)}")

    print("\n-- everything the driver exposes --")
    for attr in sorted(dir(p)):
        if attr.startswith("_"):
            continue
        try:
            print(f"  {attr:34}: {getattr(p, attr)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
