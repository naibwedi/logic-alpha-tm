# Contributing

Thank you for improving LogicAlpha. Contributions should preserve the project's
central requirement: every reported result must be reproducible and explicit
about information timing.

LogicAlpha studies whether interpretable Tsetlin Machine clauses can support
market strategy selection. Public writing should spell out “Tsetlin Machine” on
first use; the abbreviation “TM” may follow where repeated use improves clarity.

## Workflow

1. Open an issue describing the proposed research or implementation change.
2. Create a focused branch and include tests for changed behavior.
3. Run `python -m unittest discover -s tests -v`.
4. Run the synthetic demo and inspect its generated report.
5. Submit a pull request describing timing assumptions, data sources, costs, and
   any parameters examined.

Do not commit credentials, proprietary datasets, vendor exports, or results that
cannot legally be redistributed. Results based on real data must identify the
data vintage and the earliest availability timestamp used by every feature.
