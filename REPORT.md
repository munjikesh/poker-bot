# Weakness report (Rulebook 10.3)

**Team:** $mall blind
**Team leader:** Munjikesh N

## Summary

The duplicate hands are played back to back, so a bot can work out which hands are duplicates
of each other. We have **not** used this, and our submitted `bot.py` does not use it.

## The issue

Problem Statement 2.3 says: *"Bots are not told which hands are duplicates of each other."*

In the starter kit's `engine.py` (`Match.run`), however, the two hands of each duplicate deal are
always played one after the other:

```python
deal = (hand_id - 1) // 2   # hands 2k+1 and 2k+2 share a deal
```

The cards stay with the seat while the seats alternate. So in every even-numbered hand, each bot
holds the cards its opponent held in the hand before, and its opponent holds the cards the bot
itself held. A bot that simply remembers the previous hand therefore knows:

- its opponent's exact hole cards in every second hand (half of all hands), and
- the board cards already revealed in the previous hand, often the whole board.

This needs nothing beyond the `hand_id` and `hole_cards` fields the protocol already sends.

## Suggested fix

Play the two hands of each deal in separate bot processes, or at least in a random order that is
not adjacent. With reordering alone, a bot could still match a deal by its flop, so separate
processes are the robust fix.

If the official engine works the same way as the starter kit, we would appreciate a clarification
that exploiting this is prohibited, so that all teams compete on equal terms.

## How our bot avoids it

Between hands, `bot.py` keeps only statistics about the opponent's style (counts and averages
of actions and of hands shown at showdown). It stores no cards from earlier hands and has no
logic that depends on whether a hand number is odd or even; the hand number is used only to count
the hands left in the match.
