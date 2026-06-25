"""Лёгкие замеры производительности для ИИ-ассистента.

Все замеры включаются одним флагом (PERFORMANCE_LOG_ENABLED / настройка
performance_log_enabled). Когда выключено — никаких накладных расходов и логов.

Главное правило: время записи команды и время обработки ПОСЛЕ записи считаются
отдельно. Долгая речь пользователя не должна выглядеть как "тормоз ассистента".
"""

from __future__ import annotations

import time

# Глобальный флаг — выставляется приложением на старте.
ENABLED = False


def set_enabled(enabled: bool) -> None:
    global ENABLED
    ENABLED = bool(enabled)


def now() -> float:
    """Монотонные секунды для измерения интервалов."""
    return time.perf_counter()


def log(label: str, ms: float | None = None) -> None:
    """Печатает строку вида '[Perf] <label>: <ms> ms' (или без ms)."""
    if not ENABLED:
        return
    if ms is None:
        print(f"[Perf] {label}", flush=True)
    else:
        print(f"[Perf] {label}: {ms:.0f} ms", flush=True)


def log_since(label: str, start: float) -> None:
    """Печатает интервал от start до текущего момента в мс."""
    if not ENABLED:
        return
    print(f"[Perf] {label}: {(now() - start) * 1000:.0f} ms", flush=True)
