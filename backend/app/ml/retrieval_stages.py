class AlwaysInjectGate:
    """Baseline. Every chat message gets memory context."""

    def should_inject(self, query: str, query_vec, candidates: list[dict]) -> bool:
        return True


class KeepOrderRanker:
    """Baseline preserving the vector candidate order."""

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        return candidates
