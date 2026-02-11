import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
import os

st.set_page_config(layout="wide", page_title="NUTS3 Industry Profiles Viewer")

st.title("Interactive NUTS3 Industry Profiles")

# 1. File Selection
files = sorted(glob.glob("results/Industry_Regio_*_ts.csv"))
selected_file = st.sidebar.selectbox("Select Scenario File", files, index=0)

FFE_FILE = "input/external/ffe_industry_2019.csv"
CONVERTED_TS_FILE = "results/converted_timeseries_2018.csv"
CORR_FILE = "results/correlation_analysis_2023.csv"
CORR_FILE_FALLBACK = "results/correlation_analysis_2023_new.csv"

@st.cache_data
def load_data(path, mtime):
    # Including mtime in arguments ensures cache invalidates when file changes
    df = pd.read_csv(path, sep=';', decimal=',', index_col=0)
    # Create datetime index
    df.index = pd.date_range(start='2018-01-01 00:00', periods=len(df), freq='h')
    return df

def convert_2019_to_2018_smart(df_2019):
    """
    Transforms 2019 profiles to 2018 calendar.
    1. Aligns weekdays (2018-01-01 Mon <- 2019-01-07 Mon).
    2. Patches Fixed Holidays.
    3. Patches Variable Holidays.
    """
    # FIX: The FfE data has a 2-day lag (Monday behaves like Saturday).
    # Shift it by -48 hours to align with the real 2019 calendar.
    # We use roll to wrap around correctly.
    df_2019_values = np.roll(df_2019.values, -48, axis=0)
    df_2019 = pd.DataFrame(df_2019_values, index=df_2019.index, columns=df_2019.columns)

    # 1. Weekday Alignment
    # 2019-01-07 is Mon. 2018-01-01 is Mon.
    # Take 2019 data from Day 6 (Jan 7) to End.
    # Append 2019 data from Day 1 (Jan 2) to Day 6 (Jan 7) to fill the end.
    
    # 2019 has 8760 hours.
    # Start index of Jan 7 = 6 * 24 = 144
    idx_jan7 = 144
    # End of Jan 7 is not needed, we need start of Jan 2.
    # Jan 2 start = 1 * 24 = 24.
    
    # Main body: Jan 7 (00:00) to Dec 31 (23:00)
    # Length: 8760 - 144 = 8616 hours.
    part1 = df_2019.iloc[idx_jan7:].values
    
    # Tail: Jan 2 (00:00) to Jan 7 (23:00)? 
    # We need to fill 365 days total.
    # Current part1 is 359 days. We need 6 days.
    # Jan 2, 3, 4, 5, 6, 7? No, Jan 2,3,4,5,6,7 is 6 days.
    # Jan 2 (00:00) to Jan 7 (23:00) is 6 days.
    # Wait. Jan 2, 3, 4, 5, 6, 7.
    # Jan 7 is Mon. We used it for Jan 1.
    # The last day of 2018 is Dec 31 (Mon).
    # The last day of part1 is Dec 31 2019 (Tue).
    # This logic is slightly loose on the exact wrap, but consistent with preserving M-T-W-T-F-S-S cycle.
    # Let's take Jan 2 to Jan 7 (exclusive of Jan 8).
    # Jan 2 starts at 24. Jan 8 starts at 24 + 6*24 = 168.
    # Length 144.
    part2 = df_2019.iloc[24:168].values
    
    aligned_values = np.concatenate([part1, part2], axis=0)
    
    # Create 2018 DataFrame
    df_2018 = pd.DataFrame(aligned_values, columns=df_2019.columns)
    df_2018.index = pd.date_range(start='2018-01-01 00:00', end='2018-12-31 23:00', freq='h')
    
    # Helper to copy day
    def patch_day(d18, d19):
        try:
            # Check if d19 exists in source (it should)
            if d19 in df_2019.index.date: # This check is tricky with timestamp index
                # String based lookup
                src = df_2019.loc[d19].values
                df_2018.loc[d18] = src
            else:
                # Try string lookup directly
                src = df_2019.loc[d19].values
                df_2018.loc[d18] = src
        except Exception as e:
            print(f"Warning: Failed to patch {d18} from {d19}: {e}")

    # 2. Fixed Holidays
    # We want Sunday profiles for holidays. 
    # In the shifted 2019 data, Sundays are Sundays.
    fixed_holidays = [
        ('2018-01-01', '2019-01-06'), # New Year -> Jan 6 (Sun)
        ('2018-05-01', '2019-05-05'), # May 1 -> May 5 (Sun)
        ('2018-10-03', '2019-10-06'), # Unity Day -> Oct 6 (Sun)
        ('2018-10-31', '2019-11-03'), # Reformation -> Nov 3 (Sun)
        ('2018-11-01', '2019-11-03'), # All Saints -> Nov 3 (Sun)
        ('2018-12-24', '2019-12-22'), # Christmas Eve -> Dec 22 (Sun)
        ('2018-12-25', '2019-12-22'), # Christmas -> Dec 22 (Sun)
        ('2018-12-26', '2019-12-29'), # Boxing Day -> Dec 29 (Sun)
        ('2018-12-31', '2019-12-29')  # New Year's Eve -> Dec 29 (Sun)
    ]
    for d18, d19 in fixed_holidays:
        patch_day(d18, d19)
        
    # 3. Variable Holidays
    # All these should also use Sunday profiles.
    var_holidays = [
        ('2018-03-30', '2019-04-14'), # Good Friday -> nearby Sunday
        ('2018-04-02', '2019-04-14'), # Easter Monday -> nearby Sunday
        ('2018-05-10', '2019-05-12'), # Ascension -> nearby Sunday
        ('2018-05-21', '2019-05-19'), # Pentecost Monday -> nearby Sunday
        ('2018-05-31', '2019-06-02')  # Corpus Christi -> nearby Sunday
    ]
    for d18, d19 in var_holidays:
        patch_day(d18, d19)
        
    return df_2018

@st.cache_data
def load_ffe_data():
    if os.path.exists(FFE_FILE):
        path = FFE_FILE
    elif os.path.exists(FFE_FILE + ".gz"):
        path = FFE_FILE + ".gz"
    else:
        return None

    try:
        # pandas automatically handles compression='gzip' if extension is .gz
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Pivot to wide format: index=timestamp, columns=nuts3_code, values=load_kw
        df_wide = df.pivot(index='timestamp', columns='nuts3_code', values='load_kw')
        
        # Apply 2019 -> 2018 Transformation
        df_2018 = convert_2019_to_2018_smart(df_wide)
        
        return df_2018
    except Exception as e:
        st.error(f"Error loading FfE data: {e}")
        return None

@st.cache_data
def load_converted_timeseries():
    if not os.path.exists(CONVERTED_TS_FILE):
        return None
    try:
        df = pd.read_csv(CONVERTED_TS_FILE, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Error loading converted time series: {e}")
        return None

@st.cache_data
def load_correlation_analysis():
    if not os.path.exists(CORR_FILE):
        return None, None
    try:
        df = pd.read_csv(CORR_FILE, index_col="nuts3_id")
    except Exception as e:
        st.error(f"Error loading correlation analysis: {e}")
        return None, None

    fallback = None
    if os.path.exists(CORR_FILE_FALLBACK):
        try:
            fallback = pd.read_csv(CORR_FILE_FALLBACK, index_col="nuts3_id")
        except Exception:
            fallback = None

    return df, fallback

@st.cache_data
def load_metadata():
    try:
        # Load NUTS3 names (assuming no header, Col 0 = Code, Col 1 = Name)
        meta = pd.read_excel("input/other/nuts3_destatis.xlsx", header=None)
        # Create dictionary: Code -> Name
        return dict(zip(meta[0], meta[1]))
    except Exception as e:
        st.warning(f"Could not load NUTS3 names: {e}")
        return {}

if selected_file:
    # Get last modification time to track updates
    file_mtime = os.path.getmtime(selected_file)
    df = load_data(selected_file, file_mtime)
    
    # Load metadata
    nuts3_names = load_metadata()
    
    # Load methodology stats
    @st.cache_data
    def load_method_stats():
        try:
            df = pd.read_csv("results/nuts3_methodology_stats.csv", sep=';', decimal=',', index_col='nuts3_id')
            return df
        except Exception as e:
            # st.warning(f"Could not load methodology stats: {e}")
            return None
    
    method_stats = load_method_stats()
    
    # 2. Sidebar Controls
    st.sidebar.header("Filter Options")
    
    # Date Range
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
        st.stop()
        
    # Region Selection
    # Sort regions by total energy (descending) to make finding big ones easier
    sums = df.sum().sort_values(ascending=False)
    sorted_codes = sums.index.tolist()
    
    # Create display names mapping
    # format: "Name_Code"
    display_map = {}
    for code in sorted_codes:
        name = nuts3_names.get(code, "Unknown")
        display_name = f"{name}_{code}"
        display_map[display_name] = code
    
    # Default: 2 highest rep-profile share (highest consumption), 2 lowest rep-profile share (highest consumption)
    default_codes = []
    if method_stats is not None and 'rep_profile_weight' in method_stats.columns:
        try:
            ms = method_stats['rep_profile_weight'].dropna()
            # Intersect with available regions and sort by consumption within each group
            available = [c for c in sorted_codes if c in ms.index]
            if available:
                high_share = ms.loc[available].sort_values(ascending=False)
                low_share = ms.loc[available].sort_values(ascending=True)

                high_candidates = [c for c in high_share.index if c in sorted_codes]
                low_candidates = [c for c in low_share.index if c in sorted_codes]

                high_top = sorted(high_candidates[:10], key=lambda c: sums[c], reverse=True)[:2]
                low_top = sorted(low_candidates[:10], key=lambda c: sums[c], reverse=True)[:2]

                default_codes = list(dict.fromkeys(high_top + low_top))
        except Exception:
            default_codes = []

    if not default_codes:
        default_codes = sorted_codes[:4] if len(sorted_codes) >= 4 else sorted_codes
    default_display = [k for k, v in display_map.items() if v in default_codes]
    
    selected_display_names = st.sidebar.multiselect(
        "Select NUTS3 Regions (Ordered by Annual Consumption)", 
        list(display_map.keys()),
        default=default_display
    )
    
    # Map back to codes
    selected_regions = [display_map[name] for name in selected_display_names]
    
    show_rep_profile = st.sidebar.checkbox("Show Representative Profile", value=False)
    show_unclipped_rep_profile = st.sidebar.checkbox("Show Unclipped Rep. Profile", value=False)
    
    # Check if FfE data is available
    ffe_available = os.path.exists(FFE_FILE) or os.path.exists(FFE_FILE + ".gz")
    show_ffe = False
    if ffe_available:
        show_ffe = st.sidebar.checkbox("Compare with FfE (2019)", value=True)
    else:
        st.sidebar.caption("FfE Comparison unavailable (run scripts/import_ffe_data.py)")

    # Converted 2018 time series (prices / generation)
    converted_available = os.path.exists(CONVERTED_TS_FILE)
    show_converted = False
    selected_converted_cols = []
    if converted_available:
        show_converted = st.sidebar.checkbox("Show SMARD ts)", value=False)
        if show_converted:
            converted_df = load_converted_timeseries()
            if converted_df is not None and not converted_df.empty:
                default_cols = [c for c in ["Price_Price", "Reg_RE_Agg"] if c in converted_df.columns]
                selected_converted_cols = st.sidebar.multiselect(
                    "Converted Series",
                    list(converted_df.columns),
                    default=default_cols
                )
            else:
                st.sidebar.caption("SMARD time series is empty or failed to load.")
    else:
        st.sidebar.caption("SMARD time series unavailable (run newcorrelationanalysis.py)")

    # Correlation Analysis
    corr_available = os.path.exists(CORR_FILE)
    show_corr = False
    if corr_available:
        show_corr = st.sidebar.checkbox("Show Correlation Analysis", value=False)
    else:
        st.sidebar.caption("Correlation analysis unavailable (run correlation analysis)")
        
    normalize = st.sidebar.checkbox("Normalize Profiles (Factor of Mean)", value=True)
    
    # 3. Filter Data
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    df_filtered = df.loc[mask]
    
    # Load FfE data if needed
    ffe_df = None
    if show_ffe:
        ffe_df = load_ffe_data()
    
    if selected_regions or show_rep_profile or show_unclipped_rep_profile or show_converted:
        fig = go.Figure()
        
        if show_converted:
            converted_df = load_converted_timeseries()
        corr_df = None
        corr_fallback_df = None
        if show_corr:
            corr_df, corr_fallback_df = load_correlation_analysis()

        # Helper to load rep profile
        def load_rep_profile(filename):
            try:
                rep_df = pd.read_csv(filename, sep=';', decimal=',', index_col=0)
                rep_df.index = pd.to_datetime(rep_df.index, format='ISO8601')
                rep_mask = (rep_df.index.date >= start_date) & (rep_df.index.date <= end_date)
                return rep_df.loc[rep_mask, 'normalized_load']
            except Exception as e:
                st.error(f"Could not load {filename}: {e}")
                return None

        # Load Representative Profile if needed
        rep_series = None
        if show_rep_profile:
            rep_series = load_rep_profile("results/representative_profile_hourly.csv")

        # Load Unclipped Representative Profile if needed
        unclipped_rep_series = None
        if show_unclipped_rep_profile:
             unclipped_rep_series = load_rep_profile("results/representative_profile_unclipped_hourly.csv")
        
        stats_data = []
        y_axis_label = "Normalized Load (p.u.)" if normalize else "Load (kW)"
        for region in selected_regions:
            series = df_filtered[region]
            if normalize:
                mean_val = series.mean()
                if mean_val != 0:
                    series = series / mean_val
                else:
                    y_axis_label = "Load (kW)" # Fallback
            else:
                y_axis_label = "Load (kW)"
            
            # Use name from metadata if available
            region_name = nuts3_names.get(region, region)
            legend_label = f"{region_name} ({region}) - {sums[region]/1e6:.1f} GWh"
                
            fig.add_trace(go.Scatter(x=df_filtered.index, y=series, mode='lines', name=legend_label))
            
            # FfE Comparison & Metrics
            ffe_corr = None
            ffe_rmse = None
            
            if show_ffe and ffe_df is not None and region in ffe_df.columns:
                ffe_series_full = ffe_df[region]
                
                # Align FfE to current mask (position-based)
                if len(ffe_df) == len(df):
                     ffe_series = ffe_series_full.loc[mask] # Use boolean array
                     
                     # Normalize FfE if needed
                     if normalize:
                         ffe_mean = ffe_series.mean()
                         if ffe_mean != 0:
                             ffe_series = ffe_series / ffe_mean
                     
                     # Calculate Metrics (comparing 'series' vs 'ffe_series')
                     # series is already normalized if normalize=True
                     try:
                         # Ensure valid data
                         valid = series.notna() & ffe_series.notna()
                         if valid.any():
                             s1 = series[valid]
                             s2 = ffe_series[valid]
                             ffe_corr = s1.corr(s2)
                             mse = ((s1 - s2) ** 2).mean()
                             ffe_rmse = mse ** 0.5
                     except Exception as e:
                         print(f"Error calculating metrics for {region}: {e}")

                     # Calculate FfE GWh for legend
                     ffe_gwh = ffe_series_full.sum() / 1e6
                     ffe_legend = f"FfE 2019 ({region}) - {ffe_gwh:.1f} GWh"
                     
                     fig.add_trace(go.Scatter(
                         x=df_filtered.index, 
                         y=ffe_series,
                         mode='lines',
                         name=ffe_legend,
                         line=dict(dash='dot')
                     ))
            
            # Store stats for table
            total_kwh = df[region].sum()
            peak_kw = df[region].max()
            flh = total_kwh / peak_kw if peak_kw > 0 else 0
            
            rep_share_str = "N/A"
            if method_stats is not None and region in method_stats.index:
                share = method_stats.loc[region, 'rep_profile_weight']
                rep_share_str = f"{share*100:.1f}%"
            
            row_stats = {
                "Region": region,
                "Annual Energy (GWh)": f"{total_kwh/1e6:.2f}",
                "Peak Load (kW)": f"{peak_kw:.0f}",
                "FLH (h)": f"{flh:.0f}",
                "Rep. Profile Share": rep_share_str
            }
            
            if show_ffe:
                row_stats["FfE Corr"] = f"{ffe_corr:.2f}" if ffe_corr is not None else "N/A"
                row_stats["FfE RMSE"] = f"{ffe_rmse:.2f}" if ffe_rmse is not None else "N/A"

            if show_corr and corr_df is not None:
                def get_corr_value(df, col):
                    if df is None or region not in df.index or col not in df.columns:
                        return None
                    return df.loc[region, col]

                corr_pv = get_corr_value(corr_df, "corr_PV")
                corr_price = get_corr_value(corr_df, "corr_Price")
                corr_re_agg = get_corr_value(corr_df, "corr_RE_Agg")
                corr_wind_comb = get_corr_value(corr_df, "corr_Wind_Comb")

                if corr_wind_comb is None and corr_fallback_df is not None:
                    corr_wind_comb = get_corr_value(corr_fallback_df, "corr_Wind_Comb")

                row_stats["Corr PV"] = f"{corr_pv:.2f}" if corr_pv is not None else "N/A"
                row_stats["Corr Price"] = f"{corr_price:.2f}" if corr_price is not None else "N/A"
                row_stats["Corr RE Agg"] = f"{corr_re_agg:.2f}" if corr_re_agg is not None else "N/A"
                row_stats["Corr Wind Comb"] = f"{corr_wind_comb:.2f}" if corr_wind_comb is not None else "N/A"
                
            stats_data.append(row_stats)

        # Plot Converted 2018 Time Series
        if show_converted and converted_df is not None and selected_converted_cols:
            conv_mask = (converted_df.index.date >= start_date) & (converted_df.index.date <= end_date)
            conv_filtered = converted_df.loc[conv_mask]

            for col in selected_converted_cols:
                series = conv_filtered[col]
                if normalize:
                    mean_val = series.mean()
                    if mean_val != 0:
                        series = series / mean_val
                fig.add_trace(go.Scatter(
                    x=conv_filtered.index,
                    y=series,
                    mode='lines',
                    name=f"Conv2018 {col}",
                    line=dict(dash='longdash')
                ))

            if normalize:
                y_axis_label = "Normalized Value (p.u.)"
            else:
                y_axis_label = "Value (native units)"
            
        # Plot Representative Profile
        if rep_series is not None:
            name = "Representative Profile"
            series = rep_series
            if normalize:
                # Normalize to its own mean in the visible range to match normalized region profiles
                rep_mean_visible = series.mean()
                if rep_mean_visible != 0:
                    series = series / rep_mean_visible
                name += " (p.u.)"
            else:
                # Scale it to match the mean magnitude of selected regions in the visible range
                if selected_regions:
                    avg_load_magnitude = df_filtered[selected_regions].mean().mean()
                    rep_mean_visible = series.mean()
                    if avg_load_magnitude > 0 and rep_mean_visible > 0:
                        # Correctly scale so the reference shape's mean matches the regions' mean
                        series = series * (avg_load_magnitude / rep_mean_visible)
                        name += " (Scaled to avg)"
            
            fig.add_trace(go.Scatter(x=rep_series.index, y=series, mode='lines', name=name, line=dict(dash='dash', color='black')))

        # Plot Unclipped Representative Profile
        if unclipped_rep_series is not None:
            name = "Unclipped Rep. Profile"
            series = unclipped_rep_series
            if normalize:
                rep_mean_visible = series.mean()
                if rep_mean_visible != 0:
                    series = series / rep_mean_visible
                name += " (p.u.)"
            else:
                if selected_regions:
                    avg_load_magnitude = df_filtered[selected_regions].mean().mean()
                    rep_mean_visible = series.mean()
                    if avg_load_magnitude > 0 and rep_mean_visible > 0:
                        series = series * (avg_load_magnitude / rep_mean_visible)
                        name += " (Scaled to avg)"
            
            fig.add_trace(go.Scatter(x=unclipped_rep_series.index, y=series, mode='lines', name=name, line=dict(dash='dot', color='red')))


        fig.update_layout(
            title="Load Profiles",
            xaxis_title="Time",
            yaxis_title=y_axis_label,
            legend_title="Regions",
            template="plotly_white",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # SMARD (Converted 2018) Stats Table
        if show_converted and converted_df is not None and selected_converted_cols:
            st.subheader("SMARD Data Statistics (Full Year)")
            smard_stats = []
            for col in selected_converted_cols:
                series = converted_df[col].dropna()
                if series.empty:
                    continue
                smard_stats.append({
                    "Series": col,
                    "Mean": f"{series.mean():.3f}",
                    "Min": f"{series.min():.3f}",
                    "Max": f"{series.max():.3f}",
                    "Std Dev": f"{series.std():.3f}",
                    "Sum": f"{series.sum():.3f}"
                })
            if smard_stats:
                st.dataframe(pd.DataFrame(smard_stats))
            else:
                st.caption("No SMARD data available for selected series.")

        # Stats Table
        if selected_regions:
            st.subheader("Selected Regions Statistics (Full Year)")
            # stats_data is already populated in the loop above
            st.dataframe(pd.DataFrame(stats_data))
            
    else:
        st.info("Please select at least one region to visualize.")
        
else:
    st.warning("No result files found in 'results/'.")
