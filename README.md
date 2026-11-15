# Pokerbots: Discard Hold'em bot

**Team:** $mall blind

The submission is the single file [`bot.py`](bot.py) (Rulebook 5.10). It is a Python 3.11 bot
that plays Discard Hold'em through the stdin/stdout protocol, using only the standard library and
numpy.

The [weakness report](REPORT.md) required by Rulebook 10.3 is in this repository.

## How the bot plays

- **Hand evaluation:** a vectorised numpy evaluator for 5-, 6- and 7-card hands.
- **Before the flop:** a built-in table of the equity of all 22,100 three-card starting hands.
- **Reading the opponent:** a weight for every hand the opponent could hold, updated after each
  of their actions and after their exposed discard.
- **Flop:** Monte Carlo simulation (6,000 deals) chooses both the action and the card to discard.
- **Turn and river:** exact enumeration of every possible opponent hand and remaining card.
- **Choosing actions:** each candidate bet, call or fold is scored by expected chips against the
  opponent's likely hands and how they are expected to respond.
- **Adapting:** the bot learns the opponent's tendencies during the match, including how often
  they fold, how often they bluff, and whether their bet size reveals their hand.
- **Match play:** once the win is mathematically certain, the bot stops taking risks.

## Rule compliance

| Rule | Status |
|---|---|
| 5.1, 5.2: Python 3.11; standard library, numpy and scipy only | Uses the standard library and numpy |
| 5.3: 1 GB memory | About 55 MB at peak in a full 1000-hand match |
| 5.4: no files, network or processes | None used; all data is built into `bot.py` |
| 5.5, 5.6: time bank, 5 s per reply | About 40 s of 150 s per 1000-hand match; slowest reply under 0.05 s |
| 5.7: ready within 5 s | First reply after about 0.3 s |
| 5.8: only protocol messages on stdout | Debug output goes to stderr only |
| 10.1: own work | Original code; nothing copied from other teams or public bot repositories |
| 10.3: weaknesses reported, not used | See [REPORT.md](REPORT.md) |

## Running it locally

With the organisers' starter kit:

```
python3.11 engine.py bot.py example_bots/random_bot.py            # Stage 1: 400 hands
python3.11 engine.py bot.py example_bots/random_bot.py --stage 2  # Stage 2: 1000 hands
```
