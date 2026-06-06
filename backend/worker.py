import os
import time


def main() -> None:
    """
    Worker process placeholder for the cloud-ready architecture.

    EchoNotes still executes local FastAPI background tasks for MVP stability.
    This process is intentionally lightweight so Docker Compose can stand up a
    separate worker service now; later phases can move ASR/VLM/report jobs here.
    """
    queue_url = os.getenv("ECHONOTES_REDIS_URL", "redis://redis:6379/0")
    print(f"EchoNotes worker online. Queue target: {queue_url}", flush=True)
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
