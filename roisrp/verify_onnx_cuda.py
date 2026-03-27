import json
import sys
import traceback


def main() -> int:
    try:
        import onnxruntime as ort
    except Exception:
        print("FAILED_TO_IMPORT_ONNXRUNTIME")
        traceback.print_exc()
        return 2

    print(f"onnxruntime_version={ort.__version__}")

    try:
        providers = ort.get_available_providers()
    except Exception:
        print("FAILED_TO_QUERY_PROVIDERS")
        traceback.print_exc()
        return 3

    print("providers=" + json.dumps(providers, ensure_ascii=True))

    if "CUDAExecutionProvider" not in providers:
        print("CUDAExecutionProvider missing")
        return 1

    print("CUDAExecutionProvider detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
