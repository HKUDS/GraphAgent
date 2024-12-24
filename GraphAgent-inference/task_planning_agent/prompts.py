few_shot_example_1 = \
(
    """
    Input:
    Paper Title: Simple Neural Network for Image Classification
    Abstract: We present a basic neural network architecture for image classification. Our model uses three layers and achieves 85% accuracy on the test dataset. The implementation is straightforward and suitable for educational purposes. Experimental results show that this simple architecture can serve as a good baseline for more complex models.

    Which category should the paper be classified into? You have the following choices: Distributed Parallel and Cluster Computing (cs.DC), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE)
    """.rstrip()
    ,
    """
    Output:
    {{
        "knowledge_text": "Paper Title: Simple Neural Network for Image Classification. Abstract: We present a basic neural network architecture for image classification. Our model uses three layers and achieves 85% accuracy on the test dataset. The implementation is straightforward and suitable for educational purposes. Experimental results show that this simple architecture can serve as a good baseline for more complex models.",
        "task_type": "predictive",
        "user_annotation": "You have the following choices: Distributed Parallel and Cluster Computing (cs.DC), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE)"
    }}
    """.strip()
)

few_shot_example_2 = \
(
"""
Input:
Story Title: Pride and Prejudice (Chapter 1 Excerpt)
Text: Mr. Bingley was good-looking and gentlemanlike; he had a pleasant countenance, and easy, unaffected manners. His sisters were fine women, with an air of decided fashion. His brother-in-law, Mr. Hurst, merely looked the gentleman; but his friend Mr. Darcy soon drew the attention of the room by his fine, tall person, handsome features, noble mien, and the report which was in general circulation within five minutes after his entrance, of his having ten thousand a year. The gentlemen pronounced him to be a fine figure of a man, the ladies declared he was much handsomer than Mr. Bingley, and he was looked at with great admiration.
Generate a summary of the relationships and social dynamics between the characters mentioned in this text.
""".rstrip()
,
"""
Output:
{{
"knowledge_text": "Story Title: Pride and Prejudice (Chapter 1 Excerpt). Text: Mr. Bingley was good-looking and gentlemanlike; he had a pleasant countenance, and easy, unaffected manners. His sisters were fine women, with an air of decided fashion. His brother-in-law, Mr. Hurst, merely looked the gentleman; but his friend Mr. Darcy soon drew the attention of the room by his fine, tall person, handsome features, noble mien, and the report which was in general circulation within five minutes after his entrance, of his having ten thousand a year. The gentlemen pronounced him to be a fine figure of a man, the ladies declared he was much handsomer than Mr. Bingley, and he was looked at with great admiration.",
"task_type": "generative",
"user_annotation": "Generate a summary of the relationships and social dynamics between the characters mentioned in this text."
}}
""".strip()
)

few_shot_example_3 = \
(
"""
Input:
Report Title: Climate Change Impact on Arctic Wildlife (2023)
Text: The Arctic ecosystem has experienced unprecedented changes in the past decade. Polar bear populations have declined by 27% since 2013, primarily due to sea ice loss. Caribou migration patterns have shifted approximately 200km northward. Arctic foxes show increased competition with red foxes, which have expanded their range northward. Marine mammals, including seals and whales, face disrupted feeding patterns due to changing ice conditions and prey availability. Scientists project that without immediate intervention, 35% of Arctic species may face local extinction by 2050.
Synthesize the key findings and their interconnections from this report.
""".rstrip()
,
"""
Output:
{{
"knowledge_text": "Report Title: Climate Change Impact on Arctic Wildlife (2023). Text: The Arctic ecosystem has experienced unprecedented changes in the past decade. Polar bear populations have declined by 27% since 2013, primarily due to sea ice loss. Caribou migration patterns have shifted approximately 200km northward. Arctic foxes show increased competition with red foxes, which have expanded their range northward. Marine mammals, including seals and whales, face disrupted feeding patterns due to changing ice conditions and prey availability. Scientists project that without immediate intervention, 35% of Arctic species may face local extinction by 2050.",
"task_type": "generative",
"user_annotation": "Synthesize the key findings and their interconnections from this report."
}}
""".strip()
)