from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
    horizon: int = 20
    min_train: int = 504
    test_size: int = 126
    rebalance_every: int = 5
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    label_dead_zone: float = 0.003
    lambda_vol: float = 0.15
    lambda_drawdown: float = 0.20
    strategy_cost_bps: float = 2.0
    selector_switch_cost_bps: float = 2.0
    smoothing: float = 1.0
    seed: int = 7

