"""APScheduler entry point for the ingestion pipeline. Runs daily at configured time."""

import signal
import sys

import structlog
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from paper_scout.config import settings
from paper_scout.db import close, ensure_indexes
from paper_scout.pipeline import run_pipeline

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
)

log = structlog.get_logger()


def _run_job():
    """Wrapper to catch and log errors without crashing the scheduler."""
    try:
        stats = run_pipeline()
        log.info(
            "scheduled_run_complete",
            new_papers=stats.new_papers,
            candidates=stats.candidates_found,
            errors=len(stats.errors),
        )
    except Exception as e:
        log.error("scheduled_run_failed", error=str(e), exc_info=True)


def main():
    log.info(
        "scheduler_starting",
        cron=f"{settings.schedule_cron_hour}:{settings.schedule_cron_minute:02d}",
        timezone=settings.schedule_timezone,
    )

    ensure_indexes()

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _run_job,
        trigger=CronTrigger(
            hour=settings.schedule_cron_hour,
            minute=settings.schedule_cron_minute,
            timezone=settings.schedule_timezone,
        ),
        id="paper_scout_pipeline",
        name="Paper Scout Daily Pipeline",
    )

    # Also run once at startup
    log.info("running_initial_pipeline")
    _run_job()

    def _shutdown(signum, frame):
        log.info("shutdown_signal_received", signal=signum)
        scheduler.shutdown(wait=False)
        close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info(
        "scheduler_running",
        next_run=f"{settings.schedule_cron_hour}:{settings.schedule_cron_minute:02d} {settings.schedule_timezone}",
    )
    scheduler.start()


if __name__ == "__main__":
    main()
