"""Run the self-tests embedded in each module under pytest.

The modules keep their unittest suites (also runnable with ``--self-test``);
importing the TestCase classes here lets pytest and CI collect them.
"""

from eigenmind.graph_construction import Tests as GraphConstructionSelfTests  # noqa: F401
from eigenmind.selection_baselines import Tests as SelectionBaselinesSelfTests  # noqa: F401
from eigenmind.semantic_bridges import BridgeTests as SemanticBridgesSelfTests  # noqa: F401
from eigenmind.spectral_attention import SelectionTests as SpectralAttentionSelfTests  # noqa: F401
from eigenmind.supported_diversity import Tests as SupportedDiversitySelfTests  # noqa: F401
