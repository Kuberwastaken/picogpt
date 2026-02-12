#!/usr/bin/env python3
import argparse
import base64
import os
import sys
import zlib

import qrcode


def build_restore_command(input_file_name: str, b64_payload: str) -> str:
    return (
        "python -c \"import base64,zlib,pathlib;"
        f"pathlib.Path('{input_file_name}').write_bytes("
        f"zlib.decompress(base64.b64decode('{b64_payload}'),31));"
        f"print('wrote {input_file_name}')\""
    )


def build_raw_payload(input_file_name: str, b64_payload: str) -> str:
    return f"PICOQR|{input_file_name}|gzip+base64|{b64_payload}"


def make_qr(payload: str, output_file: str, error_level: str) -> None:
    levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }

    qr = qrcode.QRCode(
        version=None,
        error_correction=levels[error_level],
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a QR code that carries a compressed file payload."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="picogpt.py",
        help="File to encode (default: picogpt.py)",
    )
    parser.add_argument(
        "output_png",
        nargs="?",
        default="picogpt_qr.png",
        help="Output QR image path (default: picogpt_qr.png)",
    )
    parser.add_argument(
        "--mode",
        choices=["python-cmd", "raw"],
        default="python-cmd",
        help="Payload format inside the QR (default: python-cmd)",
    )
    parser.add_argument(
        "--ec",
        choices=["L", "M", "Q", "H"],
        default="L",
        help="QR error correction level (default: L, max capacity)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    with open(args.input_file, "rb") as f:
        raw_data = f.read()

    compressor = zlib.compressobj(level=9, wbits=31)
    compressed = compressor.compress(raw_data) + compressor.flush()
    b64_payload = base64.b64encode(compressed).decode("ascii")

    input_file_name = os.path.basename(args.input_file)
    if args.mode == "python-cmd":
        payload = build_restore_command(input_file_name, b64_payload)
    else:
        payload = build_raw_payload(input_file_name, b64_payload)

    print(f"Input bytes: {len(raw_data)}")
    print(f"Compressed bytes (gzip): {len(compressed)}")
    print(f"Payload chars: {len(payload)}")

    try:
        make_qr(payload, args.output_png, args.ec)
    except ValueError as e:
        print(f"QR generation failed: {e}")
        print("Tip: use --ec L for max capacity, or switch to a smaller file.")
        sys.exit(2)

    print(f"QR code saved to: {args.output_png}")
    if args.mode == "python-cmd":
        print("Scan/copy payload and run it in a terminal to recreate the file.")
    else:
        print("Raw mode payload is: PICOQR|<filename>|gzip+base64|<data>")


if __name__ == "__main__":
    main()