
keywords_from_scaffold_text_example = \
(
    """
    Input:
        The movie "Inception" directed by Christopher Nolan is a mind-bending thriller that explores the concept of dream invasion. The film features a complex narrative structure, stunning visual effects, and a memorable score by Hans Zimmer. The performances by Leonardo DiCaprio and the supporting cast are highly praised. However, some viewers find the plot confusing and difficult to follow.

    Output:
        {{
        "keywords": [
        {{
        "keyword_id": 0,
        "name": "director",
        "description": "The movie 'Inception' directed by Christopher Nolan."
        }},
        {{
        "keyword_id": 1,
        "name": "concept",
        "description": "Explores the concept of dream invasion."
        }},
        {{
        "keyword_id": 2,
        "name": "visual effects",
        "description": "The film features stunning visual effects."
        }},
        {{
        "keyword_id": 3,
        "name": "score",
        "description": "A memorable score by Hans Zimmer."
        }},
        {{
        "keyword_id": 4,
        "name": "performances",
        "description": "The performances by Leonardo DiCaprio and the supporting cast are highly praised."
        }},
        {{
        "keyword_id": 5,
        "name": "plot",
        "description": "Some viewers find the plot confusing and difficult to follow."
        }}
        ]
        }}
    """.rstrip()
    ,
    """
    Input:
        The movie "The Matrix" directed by the Wachowskis is a groundbreaking science fiction film that explores the nature of reality and human consciousness. The film is known for its innovative special effects, particularly the use of "bullet time" slow-motion effects. Keanu Reeves stars as Neo, a hacker who discovers the truth about the simulated reality known as the Matrix. The film's philosophical themes and action sequences have made it a cult classic.

    Output:
        {{
        "keywords": [
        {{
        "keyword_id": 0,
        "name": "director",
        "description": "The movie 'The Matrix' directed by the Wachowskis."
        }},
        {{
        "keyword_id": 1,
        "name": "concept",
        "description": "Explores the nature of reality and human consciousness."
        }},
        {{
        "keyword_id": 2,
        "name": "special effects",
        "description": "Known for its innovative special effects, particularly the use of 'bullet time' slow-motion effects."
        }},
        {{
        "keyword_id": 3,
        "name": "main character",
        "description": "Keanu Reeves stars as Neo, a hacker who discovers the truth about the simulated reality known as the Matrix."
        }},
        {{
        "keyword_id": 4,
        "name": "themes",
        "description": "The film's philosophical themes and action sequences have made it a cult classic."
        }}
        ]
        }}
    """.rstrip()
)

scaffold_texts_from_input_example = \
(
    """
    Knowledge Text:
        "Product Review: XYZ Smartwatch
        Review: The XYZ Smartwatch is a versatile and feature-rich device that offers a range of functionalities for fitness enthusiasts and tech-savvy users. It comes with a sleek design, a high-resolution display, and a variety of sensors to track your health metrics. The watch supports multiple sports modes, GPS tracking, and has a long battery life. However, some users have reported issues with the accuracy of the heart rate monitor and occasional software glitches. Overall, the XYZ Smartwatch provides good value for its price, with its comprehensive set of features and stylish design making it a popular choice among consumers.",

    Scaffold Nodes: [{{"id": 0, "type": "product", "name": "XYZ Smartwatch"}}, {{id: 1, "type": "features", "name": "features"}}, {{id: 2, "type": "performance", "name": "performance"}}, {{"id": 3, "type": "user_experience", "name": "user_experience"}}, {{id: 4, "type": "conclusion", "name": "conclusion"}}],
    """.strip()
    ,
    """
    Your output:
    {{
        "scaffold_texts": [
        {{
        "node_id": 0,
        ## hint: for here you need to summarize the product from the review ##
        "text": "The XYZ Smartwatch is a versatile and feature-rich device designed for fitness enthusiasts and tech-savvy users. It offers a sleek design, high-resolution display, and various health-tracking sensors."
        }},
        {{
        "node_id": 1,
        ## hint: for here you need to find and summarize the features ##
        "text": "The smartwatch supports multiple sports modes, GPS tracking, and has a long battery life. It also includes a variety of sensors to monitor health metrics."
        }},
        {{
        "node_id": 2,
        ## hint: for here you need to find and summarize the performance ##
        "text": "While the XYZ Smartwatch offers a range of functionalities, some users have reported issues with the accuracy of the heart rate monitor and occasional software glitches."
        }},
        {{
        "node_id": 3,
        ## hint: for here you need to find and summarize the user experience ##
        "text": "Overall, users appreciate the comprehensive set of features and stylish design of the XYZ Smartwatch, making it a popular choice among consumers."
        }},
        {{
        "node_id": 4,
        ## hint: for here you need to find and summarize the conclusion ##
        "text": "The XYZ Smartwatch provides good value for its price, with its extensive features and attractive design outweighing the reported issues, making it a recommended purchase."
        }}
        ]
    }}
    """.strip()
)