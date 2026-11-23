"""Run a finite research replay. Legacy imports remain lazy."""

from fingpt_research.cli import main


def __getattr__(name):
    if name == "TradingSystem":
        from legacy_trading import TradingSystem

        return TradingSystem
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
