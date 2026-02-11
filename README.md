# NUTS3 Industry Profile Generator

**Project Context:**
This repository contains the logic to generate hourly industrial load profiles for NUTS3 regions (Landkreise/kreisfreie Städte) in Germany. It uses a "bottom-up" approach based on real measured grid data (High Voltage and Medium Voltage) to approximate industrial behavior, filling spatial gaps with a derived "Representative Profile."

## Key Files

### 1. Main Pipeline
*   **`generate_final_nuts_profiles.py`**
    *   **Purpose:** The single entry point for the entire calculation.
    *   **Actions:**
        1.  Loads raw time series data (`input/raw/`) and spatial weights.
        2.  Calculates the **Representative Profile** (average of all valid municipalities).
        3.  Saves intermediate artifacts (unclipped/clipped representative profiles) to `results/`.
        4.  Aggregates data to NUTS3 regions, filling gaps with the representative profile. Generates both **standard** (clipped >=0) and **unclipped** (raw) profiles.
        5.  Saves gap-filling statistics to `results/nuts3_methodology_stats.csv`.
        6.  Scales final profiles to match annual GWh targets (`input/ap1_results/*_v2.xlsx`).
        7.  Outputs final CSVs to `results/` and writes `results/execution_log.txt`.

### 2. Analysis & Benchmarking
*   **`benchmark_methods.py`**
    *   **Purpose:** Evaluates 4 different mathematical formulas for calculating net industry load.
    *   **KPIs:** Counts of negative LAU/NUTS3 profiles, normalized total negative energy, and mean error severity.
    *   **Output:** `results/benchmark_results.txt`.
*   **`scripts/analysis_correlation.py`**
    *   **Purpose:** Calculates Pearson correlation coefficients between the generated NUTS3 profiles and external factors (PV generation, Wind Onshore/Offshore, and Wholesale Electricity Prices).
    *   **Logic:** Automatically aligns external 2024 data to the 2018 calendar used in the profiles.
    *   **Output:** `results/correlation_analysis_2023.csv`.

### 3. Visualization
*   **`interactive_dashboard.py`**
    *   **Purpose:** A Streamlit dashboard to inspect the generated profiles.
    *   **Usage:** `uv run streamlit run interactive_dashboard.py`
    *   **Features:** 
        *   Search regions by **Name** or **Code** (e.g., "Hamburg_DE600").
        *   Compare regions, normalize profiles, overlay the representative profile (clipped or unclipped).
        *   View detailed statistics, including Full Load Hours (FLH) and the percentage of the profile derived from the representative profile.

### 3. Mapping & Documentation
*   **`input/mapping/EU-27-LAU-2024-NUTS-2024.xlsx`**: Official Eurostat mapping used to link AGS codes to NUTS3 regions.
*   **`METHODOLOGY_DOCUMENTATION.md`**: Detailed explanation of the mathematical formulas, data sources, and logic (e.g., Option B gap filling).

### 4. Archive
*   **`archive/`**: Contains auxiliary scripts, one-off analyses, and debugging tools not required for the main pipeline.

## How to Run

1.  **Install Dependencies:** Ensure you have the required Python packages (pandas, numpy, openpyxl, streamlit, plotly).
2.  **Generate Profiles:**
    ```powershell
    python generate_final_nuts_profiles.py
    ```
    *Check `results/execution_log.txt` after running for data quality stats.*

3.  **Visualize Results:**
    ```powershell
    uv run streamlit run interactive_dashboard.py
    ```
    *You can search for regions using "Name_Code" format.*

## Data Flow Summary
`Input (Raw Excel + Weights + EU Mapping) -> [generate_final_nuts_profiles.py] -> Representative Profile (results/.npy/.csv) -> Regional Aggregation -> Scaling -> Output CSVs`

## Key Formula
**Net Industry Load (per Municipality):**
$$ P_{LAU} = P_{HS, Load} - P_{MS, Load} $$
*(Clipped to 0 if negative)*