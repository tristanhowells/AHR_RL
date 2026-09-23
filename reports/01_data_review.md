# Step 1: Data review (Betfair AUS win prices)

Source: `My Drive/racenet_data/dwbfpricesauswinDDMMYYYY.csv`. There is one file per day, and each row is one runner in a Betfair AUS horse or harness WIN market.
Scripts: `racenet/data.py` (loader and parser) and `analysis/01_data_review.py`. The full printed output is in `01_data_review_output.txt`.

## Sample analysed

The Drive folder holds about 460 daily files (Sep 2024 to Dec 2025, plus one file from Jun 2023). They can only be downloaded one at a time through the Drive connector, so this review uses **41 files**:

- A contiguous block: 8–20 Oct 2025, 8–20 Nov 2025 and 30 Nov–6 Dec 2025
- 11 scattered dates

That comes to **3,865 markets and 36,240 runners** across 149 venues. Thoroughbreds are 57% of markets and harness 43%. Every number below is from this sample, so most ROI differences are within noise. The full dataset is needed before drawing conclusions (see the end).

## What each row contains

| Field | What it is | When it's known | Usable as a model input? |
|---|---|---|---|
| `menu_hint` | Venue + date | Before the race | Yes (venue) |
| `event_name` | `R{n} {dist}m {class}` | Before the race | Yes: race no., distance, class (21 classes). Harness = `Pace*`/`Trot*` |
| `event_dt` | Scheduled off time, **UTC** | Before the race | Yes (time of day, day of week, season) |
| `selection_name` | `{saddlecloth}. {name}` | Before the race | Saddlecloth number yes. Name is only an identifier |
| `selection_id` | Betfair runner id | – | **Not a stable horse ID** (see below) |
| `bsp` | Betfair Starting Price | **At the off** | Only as a "known at the off" feature |
| `ppwap`, `ppmax`, `ppmin`, `pptradedvol` | Pre-play weighted average price, max, min, volume | **At the off** (cover the whole pre-play window) | Same caveat as BSP |
| `morningwap`, `morningtradedvol` | Documented as the morning WAP | – | **No: it leaks the result** (see below) |
| `ipmax`, `ipmin`, `iptradedvol` | In-play range and volume | After the off | No |
| `win_lose` | Result | After the off | Target |

These fields are **not present**: barrier (only saddlecloth), weight, jockey/trainer/driver, track condition, rating, form, and prize money. The folder is called `racenet_data`, but these are Betfair price files, not Racenet form. If you also have Racenet form data, that's where most of the non-price features would come from.

## Key findings

### 1. `morningwap` leaks the result. Do not use it.

- Its market book (the sum of 1/price across a race's runners) has a median of **1.39**. A genuine price book sits near 1.00–1.02.
- It predicts winners far better than BSP: multinomial log loss is **1.559 against BSP's 1.705** on the same 3,649 markets. No price set before the race can beat BSP by that much.
- `morningtradedvol` is usually larger than pre-play volume, and roughly equal to pre-play + in-play volume in about 35% of rows. So it most likely covers in-play trading.

### 2. BSP, ppwap, ppmax and ppmin behave correctly, but all of them are only final at the off

Race-level log loss, where lower is better and a uniform 1/N guess scores 2.213:

| Price | Markets | LL (price) | LL (BSP, same markets) |
|---|---|---|---|
| ppwap | 3,865 | 1.703 | 1.688 |
| ppmax | 2,865 | 1.730 | 1.711 |
| ppmin | 2,865 | 1.726 | 1.711 |

BSP is the sharpest of these prices, and ppwap tracks it closely (correlation of log prices 0.972). **The timing problem:** your planned inputs (ppmax, ppmin, ppwap, and a normalised BSP) are all finalised at the off. A backtest that feeds them to the model and then bets at BSP uses information you won't have when you place the bet. The options are:

- **(a) Research framing.** Use ppwap/ppmax/ppmin as stand-ins for the price at bet time, and exclude BSP from the inputs. This is still somewhat optimistic, because ppwap includes the last seconds of trading.
- **(b) Tradeable framing.** Build the price features at a fixed time before the off, such as T−60s, from your own recordings (the `parquet` capture schema already in this repo). Then bet at BSP.

I recommend starting with (a) to find out whether any signal exists, then confirming it with (b) before trusting any P&L.

### 3. `ppmax`/`ppmin` use a sentinel when nothing traded

`ppmax=1` with `ppmin=1001` (the same pattern as `ipmax`/`ipmin`) marks runners with no trades in the window. That's **5.5% of runners**, and **26% of markets** have at least one such runner. The loader converts these to NaN and sets a `pp_range_missing` flag. In the model, use a mask plus an indicator feature, never the raw sentinel values.

### 4. The BSP market is very efficient

- The BSP book has a median of **1.000** (5th–95th percentile: 0.946–1.050). So your "softmax-like BSP" is just BSP-implied probability normalised within the race, and the BSP overround itself carries almost no information.
- Calibration by price band (actual win rate ÷ implied probability) stays between 0.95 and 1.04 in almost every band. The exceptions are $8–11 (1.11) and $21–31 (0.86), and neither is significant at this sample size.
- Backing every runner at BSP returns **−4.35%** after 5% commission, which is roughly just the commission.

| BSP rank | n | Win % | Implied % | ROI | ± SE |
|---|---|---|---|---|---|
| 1 (fav) | 3,857 | 38.0 | 38.6 | −3.6% | 2.2% |
| 2 | 3,891 | 21.1 | 20.5 | +0.1% | 3.3% |
| 3 | 3,819 | 13.2 | 13.4 | −6.8% | 4.2% |
| 4 | 3,928 | 9.7 | 9.3 | −0.4% | 5.4% |
| 8+ | 9,623 | 1.4 | 1.6 | −10.4% | 14.0% |

No simple favourite or longshot rule beats BSP in this sample. Rank is still worth including as a feature for its interactions with other inputs. It won't be an edge by itself.

### 5. Segment ROI differences are mostly noise at this sample size

- **By race type:** harness −11.9% (±5.1%) vs thoroughbred +1.1% (±6.3%).
- **By class:** maidens +11.9% (±14.5%) and handicaps +5.4% (±10.2%).
- **By field size:** small fields (≤6) +8.4% (±15.5%) and large fields (13+) −18.8% (±8.6%).

Standard errors on all-runner ROI are large because longshot winners dominate the variance. These gaps need the full dataset, and a statistical test, before they mean anything.

### 6. Movement from ppwap to BSP

Actual-to-implied win rate stays between 0.93 and 1.03 across the firming and drifting buckets, so there's no strong signal. Both prices are final at the off anyway, so this movement isn't something you can act on with this data.

### 7. Runner identity: `selection_id` is not stable

- The same horse appears under several `selection_id`s. For example, "Butter Cup" (harness, SA) has 6 IDs across 12 runs.
- Of 19,813 runner keys (race type + name), 8,699 map to more than one ID.
- **Use `race_code|runner_name` as the horse key.** It's already built as `runner_key` in the loader. Only 13 IDs map to more than one name, so name clashes are rare. Adding a check on venue state would reduce them further.

With that key, form history can be built from the price data itself. Within this sample:

- **47%** of horses appear more than once
- The median gap between runs is **27 days** (10th–90th percentile: 6–201 days)

So days since last run, runs since last win, days since last win, win strike rate so far, and previous starting price are all computable. There are two caveats:

- The first few months form a burn-in period: every horse looks like a first starter at the start of the data.
- Coverage is limited to races Betfair listed.

### 8. Other data-quality notes

- **File date = UTC race date + 1.** The loader uses `event_dt`, not the file name.
- **Dead heats:** 8 of 3,865 markets have 2 winners. Betfair splits the stake, so staking P&L must handle them, and log-loss evaluation should use only single-winner races.
- **Header case:** older files (the 2023 one) have upper-case headers. The loader normalises them.
- **Liquidity:** median pre-play matched is **$41k per thoroughbred market and $10k per harness market**. That suggests harness markets may be less efficient but would take less stake. This matters when sizing Kelly stakes.

## What this means for the model plan

| Your planned input | Verdict |
|---|---|
| Track, distance, class, time of year | ✅ Available: venue, distance, 21 classes, race type (derived), date, off time, field size. Use embeddings for venue and class. |
| ppmax, ppmin, ppwap | ⚠️ Only final at the off (see finding 2), and ppmax/ppmin need masking (finding 3) |
| Softmax of BSP | ⚠️ Same timing issue as BSP itself. Better used as the **market prior / offset** than as an input feature |
| Favouritism rank | ✅ Easy to compute from any price. Not an edge on its own |
| BSP overround | ❌ Nearly constant (median 1.000) |
| Days since last run / win | ✅ Computable from this data using `runner_key`, after a burn-in period |
| `morningwap` | ❌ Leaks the result |

## Next steps

1. **Get the full dataset here.** Downloading about 460 files one by one through the Drive connector is slow. The quickest fix: **zip the `racenet_data` folder into a single file in Drive**, and I'll pull it in one go.
2. Build a leak-free feature table: race-level features, per-runner price features with masks, and history features computed strictly from earlier races.
3. Fit a baseline: a conditional logit on log-normalised ppwap, compared against BSP on data split by time.
4. Build the DNN: a race-level encoder plus shared runner layers, a masked softmax over the runners in each race, and a market-offset output.
5. Backtest staking: flat, full Kelly and fractional Kelly at BSP, net of commission, with dead heats handled.
