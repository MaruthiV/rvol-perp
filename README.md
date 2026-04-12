# rvol-perp

Oracle-free realized volatility perpetual instrument.

## In Plain English

Bitcoin's price swings wildly sometimes and barely moves other times. Professional traders need a simple way to bet on — or protect themselves against — how bumpy the ride will be, without trusting a middleman. That's what this project builds: a new kind of financial contract that lets you trade Bitcoin's "bumpiness" (technically called realized variance) directly on a blockchain. The key trick is that the contract figures out how volatile Bitcoin has been by just looking at the exchange's own trade history — no outside data feed needed, which means nobody can cheat by feeding it fake numbers. We built the math, wrote the code, tested it against six years of real Bitcoin data (including crashes like COVID and FTX), and discovered something important along the way: our initial computer simulations said traders could safely use 10× leverage, but when we tested against real history, 80% of those positions would have been wiped out. The honest answer turned out to be 1.5× leverage — much lower, but actually safe. We wrote an academic paper documenting the whole journey, including that failure, because getting the honest number right matters more than marketing a flashy one.
