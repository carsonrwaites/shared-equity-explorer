import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy_financial as npf
import pandas as pd

HPA_SCENARIOS = [-0.02, 0.00, 0.03, 0.05, 0.07]
BUYOUT_STRUCTURES = ['at_sale', 'forced_at_horizon']


@dataclass
class FundParams:
    fund_contribution: float = 100_000.0
    locked_growth_annual: float = 0.03      # assumed annual rate used to price buyouts
    annual_fee_pct: float = 0.01            # annual fee as % of fund's locked-price stake
    buyout_structure: str = 'at_sale'       # 'at_sale' | 'forced_at_horizon'
    buyout_horizon_yrs: int = 10            # years until forced buyout (forced_at_horizon only)
    extra_buydown_pmt: Optional[float] = None  # None → 0.0 (no buydown)
    annual_opex_per_home: float = 0.0       # flat $ operating cost per home per year
    annual_opex_pct: float = 0.001          # opex as % of fund's locked-price stake per year
    annual_capex_pct: float = 0.01          # capex as % of home value per year; fund pays its equity share
    buyout_price_basis: str = 'locked'     # 'locked' | 'market' (forced_at_horizon only)


@dataclass
class HomeParams:
    home_value: float = 300_000.0
    down_payment_pct: float = 0.10
    mortgage_rate_ann: float = 0.06
    mortgage_term_yrs: int = 30
    refi_mortgage_rate_ann: Optional[float] = None  # None → falls back to mortgage_rate_ann


@dataclass
class ScenarioResult:
    params: dict
    hpa: float
    buyout_structure: str
    schedule: pd.DataFrame       # SE path, mortgage_term_yrs * 12 rows
    conv_schedule: pd.DataFrame  # conventional baseline, same length
    exit_metrics: dict


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_irr(
    cash_flows: list,
    initial_guess: float = 0.005,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Optional[float]:
    """
    Newton-Raphson IRR on a list of (period_months, amount) tuples.
    Returns annualized IRR = (1 + r_monthly)^12 - 1, or None if unsolvable.
    """
    periods = np.array([cf[0] for cf in cash_flows], dtype=float)
    amounts = np.array([cf[1] for cf in cash_flows], dtype=float)

    if not (np.any(amounts > 0) and np.any(amounts < 0)):
        return None

    r = initial_guess
    for _ in range(max_iter):
        npv = np.sum(amounts / (1 + r) ** periods)
        dnpv = np.sum(-periods * amounts / (1 + r) ** (periods + 1))
        if abs(dnpv) < 1e-14:
            return None
        r_new = r - npv / dnpv
        if r_new < -0.9999:
            r_new = -0.5
        if abs(r_new - r) < tol:
            return (1 + r_new) ** 12 - 1
        r = r_new
    return (1 + r) ** 12 - 1


def irr_at_period(result: 'ScenarioResult', obs_period: int) -> Optional[float]:
    """
    Investor IRR as of obs_period. If obs_period >= exit_period, returns the
    terminal IRR. Otherwise constructs a truncated cash flow list and appends a
    hypothetical liquidation at the market value of the remaining fund stake.
    """
    exit_period = result.exit_metrics['exit_period']
    if obs_period >= exit_period:
        return result.exit_metrics['investor_irr']
    truncated = [(t, amt) for t, amt in result.exit_metrics['investor_cash_flows']
                 if t <= obs_period]
    hypothetical_exit = result.schedule.iloc[obs_period - 1]['fund_stake_market_value']
    truncated.append((obs_period, hypothetical_exit))
    return _compute_irr(truncated)


def _build_conv_schedule(home_params: HomeParams, hpa: float) -> pd.DataFrame:
    """Conventional mortgage schedule for the full mortgage term."""
    hv = home_params.home_value
    mo_rate = home_params.mortgage_rate_ann / 12
    n_periods = home_params.mortgage_term_yrs * 12
    monthly_hpa = (1 + hpa) ** (1 / 12)

    conv_principal = hv * (1 - home_params.down_payment_pct)
    conv_pmt = npf.pmt(mo_rate, n_periods, -conv_principal)

    rows = []
    balance = conv_principal
    for t in range(1, n_periods + 1):
        hv_t = hv * monthly_hpa ** t
        interest = balance * mo_rate
        principal = conv_pmt - interest
        balance = max(balance - principal, 0.0)
        rows.append({
            'period': t,
            'home_value': hv_t,
            'mortgage_balance': balance,
            'recipient_equity': hv_t - balance,
            'mo_interest': interest,
            'mo_principal': principal,
        })

    df = pd.DataFrame(rows)
    df['cumul_interest'] = df['mo_interest'].cumsum()
    return df


def _build_se_schedule(
    fund_params: FundParams,
    home_params: HomeParams,
    hpa: float,
    extra_buydown: float,
) -> tuple:
    """
    Build the SE amortization schedule and compute exit metrics.
    Returns (schedule_df, exit_metrics_dict).

    Monthly fund fee is charged as (annual_fee_pct / 12) * fund_stake_locked_price.
    Using the locked-price stake as the fee basis keeps the fee predictable and
    inflation-adjusted, independent of volatile home-price movements. When buydown
    payments are made, the fee naturally declines as equity is returned.

    Fee cash flows are included in investor_cash_flows, so IRR reflects both
    fee income and the terminal equity exit.
    """
    hv = home_params.home_value
    mo_rate = home_params.mortgage_rate_ann / 12
    refi_mo_rate = (home_params.refi_mortgage_rate_ann / 12
                    if home_params.refi_mortgage_rate_ann is not None
                    else mo_rate)
    n_periods = home_params.mortgage_term_yrs * 12
    monthly_hpa = (1 + hpa) ** (1 / 12)
    monthly_locked = (1 + fund_params.locked_growth_annual) ** (1 / 12)
    mo_fee_rate      = fund_params.annual_fee_pct / 12
    mo_opex_flat     = fund_params.annual_opex_per_home / 12
    mo_opex_pct_rate = fund_params.annual_opex_pct / 12
    mo_capex_rate    = fund_params.annual_capex_pct / 12

    fund_contribution = fund_params.fund_contribution
    fund_share = fund_contribution / hv
    horizon_mo = fund_params.buyout_horizon_yrs * 12
    structure = fund_params.buyout_structure

    se_principal = hv * (1 - home_params.down_payment_pct) - fund_contribution
    se_pmt = npf.pmt(mo_rate, n_periods, -se_principal)

    rows = []
    investor_cash_flows = [(0, -fund_contribution)]

    # Tracks the fund's locked-price balance; decremented by buydown payments
    fund_balance_locked = fund_contribution

    # State for forced_at_horizon (new mortgage after refinance)
    fund_paid_out = False
    fund_payout_amount = 0.0

    balance = se_principal
    active_pmt = se_pmt
    active_mo_rate = mo_rate

    for t in range(1, n_periods + 1):
        hv_t = hv * monthly_hpa ** t
        locked_price_t = fund_contribution * monthly_locked ** t

        # --- forced_at_horizon: refinance event ---
        if structure == 'forced_at_horizon' and t == horizon_mo and not fund_paid_out:
            interest = balance * mo_rate
            principal = active_pmt - interest
            balance = max(balance - principal, 0.0)

            fund_balance_locked *= monthly_locked  # grow for this period
            # Final month fee and costs on remaining locked stake before buyout
            mo_fund_fee = mo_fee_rate * fund_balance_locked
            pre_exit_stake_market = (fund_balance_locked / locked_price_t * fund_share * hv_t
                                     if locked_price_t > 0 else 0.0)
            mo_opex = mo_opex_flat + mo_opex_pct_rate * fund_balance_locked
            mo_capex = mo_capex_rate * pre_exit_stake_market
            mo_net_investor_cf = mo_fund_fee - mo_opex - mo_capex
            investor_cash_flows.append((t, mo_net_investor_cf))

            if fund_params.buyout_price_basis == 'market':
                fund_payout_amount = pre_exit_stake_market
            else:
                fund_payout_amount = fund_balance_locked  # reflects any pre-paid buydowns
            new_ref_principal = balance + fund_payout_amount
            remaining_term = n_periods - t
            new_pmt = npf.pmt(refi_mo_rate, remaining_term, -new_ref_principal) if remaining_term > 0 else 0.0
            fund_paid_out = True
            fund_balance_locked = 0.0
            active_pmt = new_pmt
            active_mo_rate = refi_mo_rate

            investor_cash_flows.append((t, fund_payout_amount))

            rows.append({
                'period': t,
                'home_value': hv_t,
                'mortgage_balance': new_ref_principal,
                'fund_stake_locked_price': 0.0,
                'fund_stake_market_value': 0.0,
                'recipient_equity': hv_t - new_ref_principal,
                'mo_interest': interest,
                'mo_principal': principal,
                'fund_buydown_pmt': fund_payout_amount,
                'mo_fund_fee': mo_fund_fee,
                'mo_opex': mo_opex,
                'mo_capex': mo_capex,
                'mo_net_investor_cf': mo_net_investor_cf,
            })
            balance = new_ref_principal
            continue

        # --- normal interest / principal ---
        interest = balance * active_mo_rate
        principal = active_pmt - interest
        balance = max(balance - principal, 0.0)

        # --- fund stake and buydown by structure ---
        if structure == 'forced_at_horizon' and fund_paid_out:
            stake_locked = 0.0
            stake_market = 0.0
            buydown = 0.0
        elif structure in ('at_sale', 'forced_at_horizon'):
            fund_balance_locked *= monthly_locked
            if extra_buydown > 0:
                reduction = min(extra_buydown, fund_balance_locked)
                fund_balance_locked = max(fund_balance_locked - reduction, 0.0)
                if reduction > 0:
                    investor_cash_flows.append((t, reduction))
                buydown = reduction
            else:
                buydown = 0.0
            stake_locked = fund_balance_locked
            stake_market = (fund_balance_locked / locked_price_t * fund_share * hv_t
                            if locked_price_t > 0 else 0.0)
        else:
            raise ValueError(f"Unknown buyout_structure: '{structure}'")

        # --- monthly fee on locked-price stake ---
        mo_fund_fee = mo_fee_rate * stake_locked

        # --- fund costs (opex + capex) ---
        mo_opex = mo_opex_flat + mo_opex_pct_rate * stake_locked
        mo_capex = mo_capex_rate * stake_market
        mo_net_investor_cf = mo_fund_fee - mo_opex - mo_capex
        investor_cash_flows.append((t, mo_net_investor_cf))

        recipient_equity = hv_t - balance - stake_market

        rows.append({
            'period': t,
            'home_value': hv_t,
            'mortgage_balance': balance,
            'fund_stake_locked_price': stake_locked,
            'fund_stake_market_value': stake_market,
            'recipient_equity': recipient_equity,
            'mo_interest': interest,
            'mo_principal': principal,
            'fund_buydown_pmt': buydown,
            'mo_fund_fee': mo_fund_fee,
            'mo_opex': mo_opex,
            'mo_capex': mo_capex,
            'mo_net_investor_cf': mo_net_investor_cf,
        })

    df = pd.DataFrame(rows)
    df['cumul_interest']         = df['mo_interest'].cumsum()
    df['cumul_fees']             = df['mo_fund_fee'].cumsum()
    df['cumul_opex']             = df['mo_opex'].cumsum()
    df['cumul_capex']            = df['mo_capex'].cumsum()
    df['cumul_net_investor_cf']  = df['mo_net_investor_cf'].cumsum()

    # --- exit metrics ---
    if structure == 'forced_at_horizon':
        exit_period = horizon_mo
        fund_exit_value = fund_payout_amount
    else:
        exit_period = n_periods
        fund_exit_value = df['fund_stake_market_value'].iloc[-1]

    investor_cash_flows.append((exit_period, fund_exit_value))

    exit_row = df.iloc[exit_period - 1]
    total_fees_paid    = df['mo_fund_fee'].sum()
    total_opex_paid    = df['mo_opex'].sum()
    total_capex_paid   = df['mo_capex'].sum()
    total_costs_paid   = total_opex_paid + total_capex_paid
    net_fee_after_costs = total_fees_paid - total_costs_paid

    exit_metrics = {
        'exit_period': exit_period,
        'fund_exit_value': fund_exit_value,
        'recipient_exit_equity': exit_row['recipient_equity'],
        'total_interest_paid': df['mo_interest'].sum(),
        'total_fees_paid': total_fees_paid,
        'total_opex_paid': total_opex_paid,
        'total_capex_paid': total_capex_paid,
        'total_costs_paid': total_costs_paid,
        'net_fee_after_costs': net_fee_after_costs,
        'total_fund_payments': df['fund_buydown_pmt'].sum(),
        'investor_cash_flows': investor_cash_flows,
        'investor_irr': _compute_irr(investor_cash_flows),
    }

    return df, exit_metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_schedule(
    fund_params: FundParams,
    home_params: HomeParams,
    hpa: float,
) -> ScenarioResult:
    """
    Build full SE and conventional schedules for a single HPA scenario.

    Returns a ScenarioResult with:
      - schedule: SE path (mortgage_term_yrs * 12 rows)
      - conv_schedule: conventional baseline (same length)
      - exit_metrics: summary dict including investor IRR, interest saved, and fees
    """
    hv = home_params.home_value
    mo_rate = home_params.mortgage_rate_ann / 12
    n_periods = home_params.mortgage_term_yrs * 12

    se_principal = hv * (1 - home_params.down_payment_pct) - fund_params.fund_contribution
    conv_principal = hv * (1 - home_params.down_payment_pct)
    se_pmt = npf.pmt(mo_rate, n_periods, -se_principal)
    conv_pmt = npf.pmt(mo_rate, n_periods, -conv_principal)

    extra_buydown = fund_params.extra_buydown_pmt if fund_params.extra_buydown_pmt is not None else 0.0

    conv_schedule = _build_conv_schedule(home_params, hpa)
    se_schedule, exit_metrics = _build_se_schedule(fund_params, home_params, hpa, extra_buydown)

    exit_metrics['interest_saved_vs_conv'] = (
        conv_schedule['mo_interest'].sum() - se_schedule['mo_interest'].sum()
    )
    exit_metrics['net_homeowner_savings'] = (
        exit_metrics['interest_saved_vs_conv'] - exit_metrics['total_fees_paid']
    )

    params_snapshot = {
        'fund_contribution': fund_params.fund_contribution,
        'fund_share_pct': round(fund_params.fund_contribution / hv * 100, 1),
        'locked_growth_annual': fund_params.locked_growth_annual,
        'annual_fee_pct': fund_params.annual_fee_pct,
        'annual_opex_per_home': fund_params.annual_opex_per_home,
        'annual_opex_pct': fund_params.annual_opex_pct,
        'annual_capex_pct': fund_params.annual_capex_pct,
        'buyout_structure': fund_params.buyout_structure,
        'buyout_horizon_yrs': fund_params.buyout_horizon_yrs,
        'extra_buydown_pmt': extra_buydown,
        'buyout_price_basis': fund_params.buyout_price_basis,
        'home_value': home_params.home_value,
        'down_payment_pct': home_params.down_payment_pct,
        'mortgage_rate_ann': home_params.mortgage_rate_ann,
        'mortgage_term_yrs': home_params.mortgage_term_yrs,
        'se_pmt': se_pmt,
        'conv_pmt': conv_pmt,
        'hpa': hpa,
    }

    return ScenarioResult(
        params=params_snapshot,
        hpa=hpa,
        buyout_structure=fund_params.buyout_structure,
        schedule=se_schedule,
        conv_schedule=conv_schedule,
        exit_metrics=exit_metrics,
    )


def run_scenarios(
    fund_params: FundParams = None,
    home_params: HomeParams = None,
    hpa_list: list = None,
) -> list:
    """
    Run all combinations of HPA scenarios x buyout structures.
    Returns a list of ScenarioResult objects (len = len(hpa_list) * 2).
    Does not mutate fund_params.
    """
    if fund_params is None:
        fund_params = FundParams()
    if home_params is None:
        home_params = HomeParams()
    if hpa_list is None:
        hpa_list = HPA_SCENARIOS

    results = []
    for hpa in hpa_list:
        for structure in BUYOUT_STRUCTURES:
            fp = dataclasses.replace(fund_params, buyout_structure=structure)
            results.append(build_schedule(fp, home_params, hpa))
    return results


def summarize_scenarios(results: list) -> pd.DataFrame:
    """
    One row per ScenarioResult. Suitable for parameter sweep analysis and
    Streamlit tables/charts.
    """
    rows = []
    for r in results:
        em = r.exit_metrics
        p = r.params
        rows.append({
            'hpa': r.hpa,
            'buyout_structure': r.buyout_structure,
            'fund_contribution': p['fund_contribution'],
            'fund_share_pct': p['fund_share_pct'],
            'annual_fee_pct': p['annual_fee_pct'],
            'home_value': p['home_value'],
            'se_pmt': round(p['se_pmt'], 2),
            'conv_pmt': round(p['conv_pmt'], 2),
            'mo_pmt_savings': round(p['conv_pmt'] - p['se_pmt'], 2),
            'exit_period': em['exit_period'],
            'fund_exit_value': round(em['fund_exit_value'], 2),
            'recipient_exit_equity': round(em['recipient_exit_equity'], 2),
            'total_interest_paid': round(em['total_interest_paid'], 2),
            'total_fees_paid': round(em['total_fees_paid'], 2),
            'total_opex_paid': round(em['total_opex_paid'], 2),
            'total_capex_paid': round(em['total_capex_paid'], 2),
            'total_costs_paid': round(em['total_costs_paid'], 2),
            'net_fee_after_costs': round(em['net_fee_after_costs'], 2),
            'interest_saved_vs_conv': round(em['interest_saved_vs_conv'], 2),
            'net_homeowner_savings': round(em['net_homeowner_savings'], 2),
            'total_fund_payments': round(em['total_fund_payments'], 2),
            'investor_irr': (
                round(em['investor_irr'] * 100, 2)
                if em['investor_irr'] is not None else None
            ),
        })
    return pd.DataFrame(rows)
