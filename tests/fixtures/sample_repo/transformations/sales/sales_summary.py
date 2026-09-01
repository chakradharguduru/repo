"""Sample module for testing AST-based chunking. Not real pipeline code."""


def compute_total(rows):
    """Sum the amount column across rows."""
    return sum(r["amount"] for r in rows)


def compute_average(rows):
    """Average the amount column across rows."""
    if not rows:
        return 0
    return compute_total(rows) / len(rows)


class SalesSummary:
    """Aggregates sales rows into a summary dict."""

    def __init__(self, rows):
        self.rows = rows

    def summarize(self):
        return {
            "total": compute_total(self.rows),
            "average": compute_average(self.rows),
        }
