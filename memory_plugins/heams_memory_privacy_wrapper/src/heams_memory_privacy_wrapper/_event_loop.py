import asyncio


def ensure_thread_event_loop() -> None:
    """Give the current thread a persistent event loop if it has none.

    Executor threads have no event loop by default, so any_llm's sync wrapper
    falls back to `asyncio.run(coro)` per call; the resulting loop churn
    orphans httpx client cleanup onto closed loops ("Event loop is closed").
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
