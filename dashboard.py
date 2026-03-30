import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy_financial as npf

from core_model import (
    FundParams, HomeParams,
    build_schedule, run_scenarios, summarize_scenarios,
    HPA_SCENARIOS, irr_at_period,
)

st.set_page_config(page_title="Shared Equity Fund Model", layout="wide")
st.title("Shared Equity Fund — Parameter Explorer")

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    hpa_labels = {-0.02: "-2%", 0.00: "0%", 0.03: "3%", 0.05: "5%", 0.07: "7%"}
    hpa_for_homeowner = st.selectbox(
        "HPA scenario",
        options=HPA_SCENARIOS,
        index=2,
        format_func=lambda x: hpa_labels[x],
        help="Annual home price appreciation rate used throughout the model. Affects equity "
             "accrual, the fund's market-value stake, and interest savings. The Homeowner tab "
             "displays this scenario; the Investor tab shows IRR across all scenarios simultaneously.",
    )

    st.header("Fund Parameters")

    fund_contribution = st.slider(
        "Fund contribution ($)",
        min_value=25_000, max_value=200_000, value=100_000, step=5_000,
        format="$%d",
        help="Dollar amount the fund contributes toward the home purchase. Reduces the "
             "homeowner's mortgage principal and monthly payment; the fund receives an equity "
             "stake proportional to this amount relative to the home value.",
    )
    annual_fee_pct = st.slider(
        "Annual fund fee (%)",
        min_value=0.0, max_value=3.0, value=0.75, step=0.25,
        format="%.2f%%",
        help="Annual fee charged to the homeowner, calculated as this percentage of the "
             "fund's locked-price stake each year. This is the fund's primary recurring "
             "income stream and compounds as the locked-price stake grows over time.",
    )
    num_homes = st.number_input(
        "Number of homes in fund",
        min_value=1, max_value=10_000, value=10, step=1,
        help="Number of homes in the fund portfolio. Scales aggregate investor metrics "
             "(total deployed capital, total annual income, etc.) shown in the Investor tab.",
    )
    buyout_structure = st.selectbox(
        "Buyout structure",
        options=["at_sale", "forced_at_horizon"],
        format_func=lambda x: {
            "at_sale": "At sale",
            "forced_at_horizon": "Forced at horizon",
        }[x],
        help="Determines when and how the homeowner buys back the fund's equity stake. "
             "'At sale' defers the buyout until the home is sold, at the market value of "
             "the fund's remaining share. 'Forced at horizon' requires the homeowner to "
             "refinance and buy out the fund at a set number of years.",
    )
    locked_growth_annual = st.slider(
        "Locked growth rate (%)",
        min_value=0.0, max_value=6.0, value=3.0, step=0.5,
        format="%.1f%%",
        help="Annual rate at which the fund's internally tracked (locked-price) stake grows. "
             "Affects both structures: fees are always charged on the locked-price stake. "
             "For forced_at_horizon with locked-price basis, this rate also directly sets "
             "the buyout amount at the horizon.",
    )
    if buyout_structure == "forced_at_horizon":
        buyout_horizon_yrs = st.slider(
            "Buyout horizon (years)", min_value=3, max_value=15, value=10, step=1,
            help="Years after purchase at which the homeowner must refinance into a new "
                 "mortgage that rolls in the fund buyout amount. After this point the "
                 "homeowner owns 100% of the equity.",
        )
        buyout_price_basis = st.radio(
            "Buyout pricing basis",
            options=["locked", "market"],
            format_func=lambda x: "Locked growth price" if x == "locked" else "Market value (HPA-based)",
            horizontal=True,
            help="How the fund's stake is priced at the forced buyout. 'Locked growth price' "
                 "uses the internally tracked stake value (predictable for the homeowner). "
                 "'Market value' prices the stake at its HPA-based value at the horizon — "
                 "higher for the homeowner if prices have risen, lower if they have fallen.",
        )
    else:
        buyout_horizon_yrs = 10  # unused but required by FundParams
        buyout_price_basis = "locked"

    st.header("Fund Cost Assumptions")

    annual_capex_pct = st.slider(
        "Annual capex (% of home value)",
        min_value=0.0, max_value=3.0, value=1.0, step=0.1,
        format="%.1f%%",
        help="Fund pays this % of current home value annually for capital expenditures "
             "(repairs, maintenance), proportional to its remaining equity stake. As the "
             "homeowner buys back equity, the fund's capex obligation shrinks.",
    )
    annual_opex_per_home = st.slider(
        "Annual opex — flat ($ per home)",
        min_value=0, max_value=2_000, value=0, step=50,
        format="$%d",
        help="Fixed annual operating cost per home regardless of stake size "
             "(e.g. admin, loan servicing, legal overhead).",
    )
    annual_opex_pct = st.slider(
        "Annual opex — proportional (% of fund stake)",
        min_value=0.0, max_value=1.0, value=0.10, step=0.05,
        format="%.2f%%",
        help="Variable operating cost as a % of the fund's locked-price stake per year "
             "(e.g. asset management, compliance costs that scale with exposure).",
    )

    st.header("Home Parameters")

    home_value = st.slider(
        "Home value ($)",
        min_value=150_000, max_value=750_000, value=300_000, step=25_000,
        format="$%d",
        help="Purchase price of the home. Determines the fund's equity share percentage, "
             "the homeowner's mortgage principal, and the base for home price appreciation.",
    )
    down_payment_pct = st.slider(
        "Down payment (%)",
        min_value=3.5, max_value=20.0, value=10.0, step=0.5,
        format="%.1f%%",
        help="Homeowner's down payment as a percentage of the home value. The remaining "
             "purchase price is split between the fund contribution and the homeowner's mortgage.",
    )
    mortgage_rate_ann = st.slider(
        "Mortgage rate (%)",
        min_value=3.0, max_value=10.0, value=6.0, step=0.25,
        format="%.2f%%",
        help="Annual interest rate on the homeowner's purchase mortgage. Determines the "
             "initial monthly payment and the total interest saved compared to a conventional "
             "mortgage of the same rate.",
    )
    if buyout_structure == "forced_at_horizon":
        refi_mortgage_rate_ann = st.slider(
            "Refinance rate (%)",
            min_value=3.0, max_value=10.0, value=mortgage_rate_ann, step=0.25,
            format="%.2f%%",
            help="Expected annual interest rate on the new mortgage at the forced refinance. "
                 "Defaults to the purchase rate. Raise this to model rate risk — if market "
                 "rates rise by the horizon, the post-refi payment could be significantly higher.",
        )
    else:
        refi_mortgage_rate_ann = None
    mortgage_term_yrs = st.selectbox(
        "Mortgage term (years)", options=[15, 20, 30], index=2,
        help="Total amortization period for the mortgage. For forced_at_horizon, the "
             "remaining term after the buyout is used to amortize the new (larger) balance.",
    )

    _se_principal = home_value * (1 - down_payment_pct / 100) - fund_contribution
    _conv_principal = home_value * (1 - down_payment_pct / 100)
    _mo_rate = mortgage_rate_ann / 100 / 12
    _n = mortgage_term_yrs * 12
    _max_buydown = float(
        npf.pmt(_mo_rate, _n, -_conv_principal) - npf.pmt(_mo_rate, _n, -_se_principal)
    )
    extra_buydown_pmt = st.slider(
        "Monthly equity buydown ($)",
        min_value=0.0,
        max_value=round(_max_buydown, 2),
        value=0.0,
        step=10.0,
        format="$%.0f",
        help="Optional extra monthly payment to accelerate buying back the fund's equity "
             "stake. Capped at the difference between the conventional and shared equity "
             "payments, so total monthly housing cost never exceeds the conventional amount.",
    )

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
fund_params = FundParams(
    fund_contribution=fund_contribution,
    annual_fee_pct=annual_fee_pct / 100,
    locked_growth_annual=locked_growth_annual / 100,
    buyout_structure=buyout_structure,
    buyout_horizon_yrs=buyout_horizon_yrs,
    extra_buydown_pmt=extra_buydown_pmt,
    buyout_price_basis=buyout_price_basis,
    annual_capex_pct=annual_capex_pct / 100,
    annual_opex_per_home=float(annual_opex_per_home),
    annual_opex_pct=annual_opex_pct / 100,
)
home_params = HomeParams(
    home_value=home_value,
    down_payment_pct=down_payment_pct / 100,
    mortgage_rate_ann=mortgage_rate_ann / 100,
    mortgage_term_yrs=mortgage_term_yrs,
    refi_mortgage_rate_ann=refi_mortgage_rate_ann / 100 if refi_mortgage_rate_ann is not None else None,
)

result = build_schedule(fund_params, home_params, hpa=hpa_for_homeowner)
all_results = run_scenarios(fund_params, home_params)
summary_df = summarize_scenarios(all_results)

p = result.params
em = result.exit_metrics
sched = result.schedule.copy()
conv = result.conv_schedule.copy()

# Year column for charts
sched["year"] = sched["period"] / 12
conv["year"] = conv["period"] / 12

# ---------------------------------------------------------------------------
# Observation year slider
# ---------------------------------------------------------------------------
exit_yr = em['exit_period'] // 12
obs_year = st.slider(
    "Observation year",
    min_value=1,
    max_value=mortgage_term_yrs,
    value=exit_yr,
    step=1,
    help="Slide to observe homeowner and investor metrics at any point in time. "
         "Defaults to the exit year.",
)
obs_period = min(obs_year * 12, len(sched))
obs_idx = obs_period - 1  # 0-indexed

# Observation-year derived values (used in both tabs)
interest_saved_obs = conv['cumul_interest'].iloc[obs_idx] - sched['cumul_interest'].iloc[obs_idx]
fees_obs           = sched['cumul_fees'].iloc[obs_idx]
net_savings_obs    = interest_saved_obs - fees_obs
costs_obs          = sched['cumul_opex'].iloc[obs_idx] + sched['cumul_capex'].iloc[obs_idx]
net_fee_obs        = fees_obs - costs_obs
capex_obs          = sched['cumul_capex'].iloc[obs_idx]
opex_obs           = sched['cumul_opex'].iloc[obs_idx]
at_terminal        = obs_period >= em['exit_period']
obs_label          = "" if at_terminal else f" (yr {obs_year})"

# All-in monthly cost at observation period
mo_pmt_obs     = sched['mo_interest'].iloc[obs_idx] + sched['mo_principal'].iloc[obs_idx]
mo_fee_obs     = sched['mo_fund_fee'].iloc[obs_idx]
fund_active    = sched['fund_stake_market_value'].iloc[obs_idx] > 0
mo_buydown_obs = extra_buydown_pmt if fund_active else 0.0
allin_obs      = mo_pmt_obs + mo_fee_obs + mo_buydown_obs

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_homeowner, tab_investor = st.tabs(["Homeowner View", "Investor View"])


# ============================================================
# TAB 1 — Homeowner
# ============================================================
with tab_homeowner:

    # Derived display label
    fund_share_pct = p["fund_share_pct"]
    st.markdown(
        f"<small>Fund contributes <strong>${fund_contribution:,}</strong> ({fund_share_pct:.1f}% of home value) | "
        f"HPA scenario: <strong>{hpa_labels[hpa_for_homeowner]}</strong></small>",
        unsafe_allow_html=True,
    )

    # --- Metric row ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "SE monthly payment",
        f"${p['se_pmt']:,.0f}",
        help="The homeowner's fixed mortgage payment under the shared equity structure. "
             "Lower than the conventional payment because the fund contribution reduces "
             "the mortgage principal. This payment does not include the fund fee or buydown.",
    )
    allin_delta = p['conv_pmt'] - allin_obs
    c2.metric(
        f"All-in monthly (yr {obs_year})",
        f"${allin_obs:,.0f}",
        delta=f"{'-' if allin_delta > 0 else ''}${abs(allin_delta):,.0f} vs. conv.",
        delta_color="inverse",
        help="Mortgage payment + fund fee + equity buydown at the observation year. "
             "Fee and buydown drop to zero once the fund's stake is fully paid off. "
             "Green = cheaper than conventional; red = more expensive.",
    )
    c3.metric(
        "Conv. monthly payment",
        f"${p['conv_pmt']:,.0f}",
        help="The monthly payment on a conventional mortgage for the same home with the "
             "same down payment and interest rate — no fund involvement. Used as the "
             "baseline for all savings comparisons.",
    )
    c4.metric(
        f"Interest saved{obs_label}",
        f"${interest_saved_obs:,.0f}",
        help="Cumulative mortgage interest saved vs. the conventional path to this point. "
             "Reflects the lower principal from the fund contribution. Does not account "
             "for fund fees — see Net savings for the combined picture.",
    )
    c5.metric(
        f"Net savings{obs_label}",
        f"${net_savings_obs:,.0f}",
        delta=f"-${fees_obs:,.0f} in fees",
        delta_color="off",
        help="Interest saved minus cumulative fund fees paid to this point. This is the "
             "homeowner's true financial benefit from the program. The delta shows total "
             "fees paid, which reduce gross interest savings.",
    )

    st.divider()

    # --- Chart A: Equity accrual ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Equity accrual over time")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sched["year"], sched["recipient_equity"] / 1_000,
                label="Shared equity path", color="#1f77b4", linewidth=2)
        ax.plot(conv["year"], conv["recipient_equity"] / 1_000,
                label="Conventional path", color="#aec7e8", linewidth=2, linestyle="--")
        if buyout_structure == "forced_at_horizon":
            ax.axvline(x=buyout_horizon_yrs, color="orange", linestyle=":", linewidth=1.5,
                       label=f"Horizon ({buyout_horizon_yrs} yr)")
        ax.axvline(x=obs_year, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Yr {obs_year}", alpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}k"))
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Chart B: Cumulative cost ---
    with col_b:
        st.subheader("Cumulative cost over time")
        se_total_cost = sched["cumul_interest"] + sched["cumul_fees"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sched["year"], sched["cumul_interest"] / 1_000,
                label="SE interest", color="#1f77b4", linewidth=2)
        ax.plot(sched["year"], se_total_cost / 1_000,
                label="SE interest + fees", color="#1f77b4", linewidth=2,
                linestyle="--", alpha=0.7)
        ax.plot(conv["year"], conv["cumul_interest"] / 1_000,
                label="Conv. interest", color="#aec7e8", linewidth=2, linestyle="--")
        if buyout_structure == "forced_at_horizon":
            ax.axvline(x=buyout_horizon_yrs, color="orange", linestyle=":", linewidth=1.5,
                       label=f"Horizon ({buyout_horizon_yrs} yr)")
        ax.axvline(x=obs_year, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Yr {obs_year}", alpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}k"))
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Chart C: Ownership share % ---
    st.divider()
    st.subheader("Ownership share over time (% of home value)")
    st.caption("Remainder of 100% is mortgage balance. Both paths start at the same down payment %. "
               "Monthly buydowns accelerate equity transfer from the fund to the homeowner.")

    se_equity_pct        = sched["recipient_equity"] / sched["home_value"] * 100
    conv_equity_pct      = conv["recipient_equity"]  / conv["home_value"]  * 100
    fund_share_pct_series = sched["fund_stake_market_value"] / sched["home_value"] * 100

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(sched["year"], se_equity_pct,         label="SE homeowner equity",    color="#1f77b4", linewidth=2)
    ax.plot(conv["year"],  conv_equity_pct,        label="Conv. homeowner equity", color="#aec7e8", linewidth=2, linestyle="--")
    ax.plot(sched["year"], fund_share_pct_series,  label="Fund equity share",      color="#ff7f0e", linewidth=2)

    if buyout_structure == "forced_at_horizon":
        ax.axvline(x=buyout_horizon_yrs, color="orange", linestyle=":", linewidth=1.5,
                   label=f"Horizon ({buyout_horizon_yrs} yr)")
    elif extra_buydown_pmt and extra_buydown_pmt > 0:
        buyout_rows = sched[sched["fund_stake_market_value"] <= 0]
        if not buyout_rows.empty:
            buyout_yr = buyout_rows.iloc[0]["year"]
            ax.axvline(x=buyout_yr, color="#ff7f0e", linestyle=":", linewidth=1.5,
                       label=f"Fund fully bought out (yr {buyout_yr:.1f})")
    ax.axvline(x=obs_year, color="gray", linestyle="--", linewidth=1.2,
               label=f"Yr {obs_year}", alpha=0.7)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Year")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- forced_at_horizon refinance callout ---
    if buyout_structure == "forced_at_horizon":
        st.divider()
        st.subheader(f"Refinance at year {buyout_horizon_yrs}")
        horizon_row = sched[sched["period"] == buyout_horizon_yrs * 12].iloc[0]
        post_rows = sched[sched["period"] > buyout_horizon_yrs * 12]
        if not post_rows.empty:
            new_mo_pmt = post_rows.iloc[0]["mo_interest"] + post_rows.iloc[0]["mo_principal"]
        else:
            new_mo_pmt = float("nan")

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Fund buyout amount", f"${em['fund_exit_value']:,.0f}")
        rc2.metric("New mortgage balance (post-refi)", f"${horizon_row['mortgage_balance']:,.0f}")
        rc3.metric("New monthly payment (post-refinance)", f"${new_mo_pmt:,.0f}")
        basis_label = (
            "the locked growth price" if buyout_price_basis == "locked"
            else "the market value of the fund's stake (HPA-based)"
        )
        st.caption(
            f"At the horizon, the recipient refinances into a new mortgage covering the "
            f"remaining balance plus the fund buyout amount at {basis_label}."
        )


# ============================================================
# TAB 2 — Investor
# ============================================================
with tab_investor:

    st.markdown(
        f"<small>Fund contributes <strong>&#36;{fund_contribution:,}</strong> per home | "
        f"Annual fee: <strong>{annual_fee_pct:.2f}%</strong> | "
        f"Capex: <strong>{annual_capex_pct:.1f}%</strong> of home value | "
        f"Opex: <strong>&#36;{annual_opex_per_home:,}/home</strong> + <strong>{annual_opex_pct:.2f}%</strong> of stake | "
        f"Portfolio: <strong>{num_homes:,} homes</strong></small>",
        unsafe_allow_html=True,
    )

    # --- IRR chart ---
    irr_title_suffix = "at exit" if at_terminal else f"yr {obs_year} (paper)"
    st.subheader(f"Investor IRR by HPA scenario and buyout structure — {irr_title_suffix} (net of opex & capex)")

    irr_data = {}
    for r in all_results:
        val = irr_at_period(r, obs_period)
        irr_data.setdefault(r.buyout_structure, {})[r.hpa] = round(val * 100, 2) if val is not None else None
    irr_pivot = pd.DataFrame(irr_data).reindex(HPA_SCENARIOS)
    irr_pivot.index = [hpa_labels[h] for h in irr_pivot.index]
    irr_pivot.columns = [
        {"at_sale": "At sale", "forced_at_horizon": "Forced at horizon"}[c]
        for c in irr_pivot.columns
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = range(len(irr_pivot))
    width = 0.35
    colors = ["#1f77b4", "#2ca02c"]
    for i, col in enumerate(irr_pivot.columns):
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], irr_pivot[col], width,
                      label=col, color=colors[i], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(irr_pivot.index)
    ax.set_ylabel("Annualized IRR (%)")
    ax.set_xlabel("HPA scenario")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --- Portfolio aggregates (per-home × num_homes, at observation year) ---
    st.subheader("Fund-level impact (all homes)")

    stat_view = st.radio(
        "View as",
        options=["Lifetime total", "Annual average"],
        horizontal=True,
        label_visibility="collapsed",
    )
    scale = 1.0 if stat_view == "Lifetime total" else 1.0 / obs_year
    view_suffix = "" if stat_view == "Lifetime total" else f" /yr (÷ {obs_year:.0f} yrs)"

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric(
        f"Interest saved{obs_label}{view_suffix} ({hpa_labels[hpa_for_homeowner]} HPA)",
        f"${interest_saved_obs * num_homes * scale:,.0f}",
        help="Across all homes in the fund under the selected HPA scenario."
    )
    fc2.metric(
        f"Fees collected{obs_label}{view_suffix}",
        f"${fees_obs * num_homes * scale:,.0f}",
    )
    fc3.metric(
        "Total capital deployed",
        f"${fund_contribution * num_homes:,.0f}",
        help="One-time deployment amount; not time-averaged.",
    )
    fc4.metric(
        f"Net fee after costs{obs_label}{view_suffix}",
        f"${net_fee_obs * num_homes * scale:,.0f}",
        delta=f"-${costs_obs * num_homes * scale:,.0f} fund costs",
        delta_color="off",
        help="Gross fee income minus opex and capex across all homes under the selected scenario.",
    )

    st.divider()
    st.subheader("Fund cost breakdown (all homes)")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(
        f"Capex paid by fund{obs_label}{view_suffix}",
        f"${capex_obs * num_homes * scale:,.0f}",
        help="Fund's proportional share of capital expenditures on homes.",
    )
    cc2.metric(
        f"Opex paid by fund{obs_label}{view_suffix}",
        f"${opex_obs * num_homes * scale:,.0f}",
        help="Fund operating expenses (flat per-home + proportional to stake).",
    )
    cc3.metric(
        "Cost as % of gross fees",
        f"{costs_obs / fees_obs * 100:.1f}%" if fees_obs > 0 else "N/A",
        help="Total costs as a share of gross fee income — fund's expense ratio.",
    )

    st.divider()

    # --- Summary table ---
    st.subheader("Scenario summary — all HPA × buyout structure combinations")

    display_cols = [
        "hpa", "buyout_structure", "mo_pmt_savings",
        "interest_saved_vs_conv", "total_fees_paid",
        "total_costs_paid", "net_fee_after_costs",
        "net_homeowner_savings", "investor_irr",
    ]
    display_df = summary_df[display_cols].copy()
    display_df["hpa"] = display_df["hpa"].map(hpa_labels)
    display_df["buyout_structure"] = display_df["buyout_structure"].map({
        "at_sale": "At sale",
        "forced_at_horizon": "Forced at horizon",
    })
    display_df.columns = [
        "HPA", "Buyout structure", "Mo. pmt savings ($)",
        "Interest saved ($)", "Fees paid ($)",
        "Total costs ($)", "Net fee ($)",
        "Net savings ($)", "Investor IRR (%)",
    ]
    st.dataframe(display_df, width='stretch', hide_index=True)
