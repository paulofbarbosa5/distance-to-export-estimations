## Data and replication

The raw firm-level data used in the paper are confidential and are not included in this repository. The analysis was not run through BPLIM. However, the code is designed to be compatible with the BPLIM secure environment because BPLIM provides access to the Central Balance Sheet database, which is annual firm-level data for Portuguese non-financial corporations and is mostly based on IES reporting.

Authorised researchers with access to an equivalent Central Balance Sheet/IES extract can reproduce the empirical workflow by adapting the configuration file to their approved data extract and running the scripts in sequence. Numerical results may differ slightly across data vintages or extracts.

This repository contains:
- Python scripts for panel construction, validation splits, model estimation, evaluation, bootstrap inference and figure/table generation;
- configuration templates for mapping original IES/CB variables into standardized variable names;
- documentation of required variables and outputs;
- paper-ready table and figure generation routines.

This repository does not contain:
- raw IES/CB data;
- firm identifiers;
- firm-level predictions;
- fitted models based on confidential data;
- any file that would allow firm-level disclosure.
