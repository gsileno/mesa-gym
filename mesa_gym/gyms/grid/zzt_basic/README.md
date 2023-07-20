# ZZT-like 2D grid world

This game is inspired by the old classic ZZT, one of the first shareware, whose success was the beginning of a series of events bringing up to the development of the Unity platform.

The map of the target 2D environment can be specified (and visualized) as a multi-line string.

For instance:

```
map = """
|--------------------|
|                    |
|  ██████            |
|  █             ☺   |
|  █                 |
|  ██████            |
|  ░░░░░█    ████████|
|  ░░░░░█         Ω  |
|  ██████            |
|              ♦     |
|                    |
|--------------------|
"""
```

The map shows:

- a player `☺`, 
- a lion `Ω`, which ends the game if it touches the player, 
- some grass `░`, that grows in time, but is destroyed to some degree when someone walks on it
- walls `█` (that blocks movement), a gem `♦` (the end goal).

If the player touches the gem, the game ends with a win.
If the lion touches the player, the game ends with a loss.
If the time counts ends, the game ends with no win, nor loss.

All components of the scene are agents handled by mesa.