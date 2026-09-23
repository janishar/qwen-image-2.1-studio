def main() -> None:
    """Entry point for `qwen-image-2-1` CLI."""
    # Lazy import to avoid circular import warning when run as `python -m`
    from qwen_image_2_1.generate import main as _main

    _main()
