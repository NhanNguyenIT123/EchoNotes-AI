import os


def main() -> None:
    """Run a real Redis/RQ worker for long EchoNotes jobs."""
    from redis import Redis
    from rq import Worker

    queue_url = os.getenv("ECHONOTES_REDIS_URL", "redis://redis:6379/0")
    queue_name = os.getenv("ECHONOTES_RQ_QUEUE", "echonotes")
    print(f"EchoNotes RQ worker online. Queue: {queue_name}. Redis: {queue_url}", flush=True)
    redis_conn = Redis.from_url(queue_url)
    worker = Worker([queue_name], connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
