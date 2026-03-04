"""Allow running benchmarks as a package: python -m benchmarks."""
import sys

from benchmarks.runner import main

sys.exit(main())
