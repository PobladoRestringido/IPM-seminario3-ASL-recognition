import random

WORD_BANK = [
    "APPLE",
    "BRIDGE",
    "CANDLE",
    "DRAGON",
    "EMBER",
    "FALCON",
    "GARDEN",
    "HOUSE",
    "ISLAND",
    "JUNGLE",
    "KITE",
    "LEMON",
    "MOUSE",
    "NIGHT",
    "OCEAN",
    "PIANO",
    "QUEEN",
    "RIVER",
    "STONE",
    "TIGER",
    "GINGER",
    "BIGGER",
    "DIGGER",
    "JEW",
    "NETANYAHU",
]


def pick_word(available_letters):
    # Keep only words whose letters are all recognizable
    eligible = [w for w in WORD_BANK if all(c in available_letters for c in w)]
    return random.choice(eligible) if eligible else None
