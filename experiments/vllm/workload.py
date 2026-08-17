import itertools

PROMPTS = [
    "Explain quantum entanglement to a curious 12-year-old.",
    "Write a short haiku about a rainy afternoon in Tokyo.",
    "List three creative uses for a paperclip.",
    "Describe the taste of a perfectly ripe peach.",
    "What are the main causes of the French Revolution?",
    "Summarize the plot of The Great Gatsby in three sentences.",
    "Give me a 5-step recipe for chocolate chip cookies.",
    "Explain how a transformer neural network works.",
    "Write a persuasive paragraph about protecting oceans.",
    "What is the difference between TCP and UDP?",
    "Describe the process of photosynthesis briefly.",
    "Write a limerick about a mischievous cat.",
    "What are the pros and cons of remote work?",
    "Explain the Monty Hall problem in one paragraph.",
    "Suggest a weekend itinerary in Lisbon.",
    "Describe the color blue to someone who has never seen it.",
]


def make_workload(n, prompts=None):
    return list(itertools.islice(itertools.cycle(prompts or PROMPTS), n))
