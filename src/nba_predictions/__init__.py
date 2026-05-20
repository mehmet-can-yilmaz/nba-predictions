"""NBA game-winner prediction — Math 17 group project.

Public API:

    from nba_predictions import (
        elo, logistic_regression, neural_net,
        metrics, calibration, market, features, synthetic,
    )
"""

from . import (  # noqa: F401
    calibration,
    elo,
    features,
    logistic_regression,
    market,
    metrics,
    neural_net,
    synthetic,
)

__version__ = "0.1.0"
