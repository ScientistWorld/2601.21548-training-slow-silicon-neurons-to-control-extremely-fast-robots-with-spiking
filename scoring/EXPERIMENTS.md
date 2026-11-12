# Experiment Overview

Every experiment scores 10 independently trained controllers per method. The evaluator executes them on 1000 episodes it holds itself, and reports pooled success, the weakest controller's success, the contact rate, and the measured controller size.

## randomized_air_hockey
The headline setting. Can a compact controller reliably intercept a fast puck whose launch position and speed both vary? It measures timing quality under the paper's main randomized condition with a 1020-unit budget.

## velocity_range_wide
Does a controller of the same size stay effective when the puck speed distribution is widened? This tests whether the timing adapts across a broader range of arrival times.

## velocity_range_extreme
The broadest speed range the paper discusses, where it reports a clear loss for a fixed-size controller. Here, small timing errors at the fastest and slowest launches decide the score. This is where improvements over the reference are expected to show.

## single_core_extreme
The same broad speed range with a quarter of the capacity (256 units, one neuromorphic core). It measures how much interception performance a method extracts per unit of controller capacity, which is the compactness the paper's claim rests on. Exceeding the budget fails the run.
