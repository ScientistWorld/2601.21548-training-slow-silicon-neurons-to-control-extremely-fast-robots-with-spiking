# Direction

Improve how quickly and how reliably a fixed random spiking substrate's linear
readout is learned online from reward, within 2000 trials per instance, across
the paper's launch-speed and launch-position settings.

Open: credit assignment, exploration, conditioning and normalization of the
readout update, step-size control, value or world-model learning from the
observed spike trains and rewards, replay of past episodes, and any computation
between episodes. Gains should hold across independently sampled instances: the
hidden instances are new to every method, so nothing learned about a specific
public instance transfers.

Out of scope: changing the substrate, the readout form, the episode budget or
the task; using robot states, launches or substrate parameters.
